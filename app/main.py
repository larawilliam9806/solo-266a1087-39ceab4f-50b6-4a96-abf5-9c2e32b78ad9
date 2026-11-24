"""FastAPI 入口：日报补档交单与核算查询。

并发模型
--------
* 每个演出季一把进程内互斥锁，保证同一季交单串行化；不同演出季互不阻塞。
* 交单分三阶段，尽量缩短 SQLite 写锁持有时间：

  1. **锁内短读**：只读核对演出季与"所见批号"，批号过期立刻 409，
     不占用任何写锁；
  2. **锁内核算**：运行 NumPy 全点对最长路（CPU 密集、耗时），
     此阶段不开启数据库写事务，因此：
     - 其他演出季可以正常取得写锁完成交单；
     - 核算查询走 WAL 只读连接，随时可读已生效批次；
  3. **短事务提交**：核算完成后才开 ``BEGIN IMMEDIATE``，事务内再次核对
     批号（防御进程外写入者），随后把新批次、单据、核算结论/矛盾证据
     在同一事务中原子写入并提交。批号唯一约束作为最后一道防线，
     冲突时回滚并返回 409，沿用当前生效批次。

* 核算或写入抛错时事务整体回滚，旧批次保持生效；同季锁随即释放，
  排队的后续交单仍可继续，最终仍满足"同一批号最多一笔生效，其余 409"。
"""

from __future__ import annotations

import sqlite3
import threading
from contextlib import asynccontextmanager
from datetime import date, timedelta
from typing import Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from . import database as db
from .schemas import (
    AccountingResponse,
    DayResult,
    OrderPayload,
    OrderResult,
    SeasonSummary,
    SubmissionRequest,
    SubmissionResponse,
    MAX_SEASON_DAYS,
)
from .solver import OrderInput, solve

@asynccontextmanager
async def lifespan(_: FastAPI):
    db.init_db()
    yield


app = FastAPI(
    title="剧院出票日报补档 API",
    version="1.0.0",
    description="按演出季整批交单、SQLite 存档、基于差分约束补算每日出票量上下限。",
    lifespan=lifespan,
)

_season_locks: dict[str, threading.Lock] = {}
_locks_guard = threading.Lock()


def _season_lock(season_id: str) -> threading.Lock:
    with _locks_guard:
        lock = _season_locks.get(season_id)
        if lock is None:
            lock = threading.Lock()
            _season_locks[season_id] = lock
        return lock


@app.exception_handler(HTTPException)
async def _http_exc_handler(_: Request, exc: HTTPException) -> JSONResponse:
    return JSONResponse(status_code=exc.status_code, content={"error": exc.detail})


def _day_index(start: date, d: date) -> int:
    return (d - start).days + 1  # 1 基，首尾均计入


def _validate_common(
    start: date, end: date, orders: list[OrderPayload]
) -> int:
    if end < start:
        raise HTTPException(422, "演出季结束日期不能早于开始日期")
    n_days = (end - start).days + 1
    if n_days > MAX_SEASON_DAYS:
        raise HTTPException(422, f"演出季跨度最多 {MAX_SEASON_DAYS} 天")

    seen_nos: set[str] = set()
    for o in orders:
        if o.end_date < o.start_date:
            raise HTTPException(422, f"单据 {o.order_no} 的统计结束日早于开始日")
        if not (start <= o.start_date <= end and start <= o.end_date <= end):
            raise HTTPException(422, f"单据 {o.order_no} 的统计日期超出本季起止范围")
        if o.order_no in seen_nos:
            raise HTTPException(422, f"单据号 {o.order_no} 在本批中重复")
        seen_nos.add(o.order_no)
    return n_days


def _check_optimistic_batch(
    conn, season_id: str, payload: SubmissionRequest, n_days: int
) -> tuple[Optional[sqlite3.Row], int]:
    """核对演出季与所见批号（只读，不持有写锁）。

    返回 (season 行, 当前批号)；批号冲突/交单前提不成立时直接抛 409/422。
    """
    season = db.get_season(conn, season_id)
    if season is None:
        if payload.seen_batch is not None:
            raise HTTPException(
                409,
                "本季尚无任何生效批次，首次交单不能携带 seen_batch",
            )
        return None, 0

    if (
        season["start_date"] != payload.start_date.isoformat()
        or season["end_date"] != payload.end_date.isoformat()
    ):
        raise HTTPException(
            422,
            "本季起止日与已存档的演出季不一致，"
            f"存档范围：{season['start_date']} ~ {season['end_date']}",
        )
    current = db.current_batch_no(conn, season_id)
    if payload.seen_batch is None:
        raise HTTPException(
            409,
            f"本季已有生效批次 {current}，再次交单必须携带 seen_batch",
        )
    if payload.seen_batch != current:
        raise HTTPException(
            409,
            f"批号已过期：所见批号 {payload.seen_batch}，"
            f"当前生效批号 {current}，本次交单未生效，沿用当前批",
        )
    return season, current


@app.post("/api/seasons/submissions", response_model=SubmissionResponse)
def submit(payload: SubmissionRequest) -> SubmissionResponse:
    n_days = _validate_common(payload.start_date, payload.end_date, payload.orders)
    lock = _season_lock(payload.season_id)

    # 同季串行化；不同季各用各的锁，互不阻塞。锁内绝不长时间持有 DB 写锁。
    with lock:
        # ---- 阶段 1：短读核对（只读连接，不拿写锁）----
        conn = db.connect()
        try:
            season, current = _check_optimistic_batch(
                conn, payload.season_id, payload, n_days
            )
        finally:
            conn.close()

        # ---- 阶段 2：锁内核算（CPU 密集，不开启任何写事务）----
        # 此期间其他季可正常交单，核算查询也始终可读已生效批次。
        order_inputs = [
            OrderInput(
                order_no=o.order_no,
                l=_day_index(payload.start_date, o.start_date),
                r=_day_index(payload.start_date, o.end_date),
                total=o.total,
            )
            for o in payload.orders
        ]
        result = solve(n_days, order_inputs)
        new_batch = current + 1
        order_rows = [
            (o.order_no, oi.l, oi.r, oi.total)
            for o, oi in zip(payload.orders, order_inputs)
        ]
        daily_rows = [
            (d.day_index, d.min_tickets, d.max_tickets) for d in result.days
        ]

        # ---- 阶段 3：短事务，复检批号后整批原子写入 ----
        conn = db.connect()
        try:
            with db.transaction(conn):
                # 防御进程外写入者：事务内再核对一次批号。
                recheck_season = db.get_season(conn, payload.season_id)
                if recheck_season is None:
                    # 季在阶段 1 后被外部清理（常规部署不会发生），按首交补建。
                    if season is not None:
                        raise HTTPException(
                            409,
                            "本季存档状态已变化，本次交单未生效，请重新查询批号",
                        )
                    db.insert_season(
                        conn,
                        payload.season_id,
                        payload.start_date,
                        payload.end_date,
                        n_days,
                    )
                else:
                    current_now = db.current_batch_no(conn, payload.season_id)
                    if current_now != current:
                        raise HTTPException(
                            409,
                            f"批号已过期：所见批号 {payload.seen_batch}，"
                            f"当前生效批号 {current_now}，本次交单未生效，沿用当前批",
                        )

                try:
                    db.insert_batch(
                        conn, payload.season_id, new_batch, result.feasible
                    )
                    db.insert_orders(conn, payload.season_id, new_batch, order_rows)
                    if result.feasible:
                        db.insert_daily_bounds(
                            conn, payload.season_id, new_batch, daily_rows
                        )
                    else:
                        db.insert_contradictions(
                            conn,
                            payload.season_id,
                            new_batch,
                            result.contradiction_proof,
                        )
                except sqlite3.IntegrityError as exc:
                    # UNIQUE(season_id, batch_no) 等约束兜底：
                    # 同一批号并发写入时，输家回滚并按批号冲突返回 409。
                    raise HTTPException(
                        409,
                        f"批号 {new_batch} 已被其他交单占用，本次交单未生效，沿用当前批",
                    ) from exc

            return SubmissionResponse(
                season_id=payload.season_id,
                batch_no=new_batch,
                feasible=result.feasible,
                n_orders=len(payload.orders),
                contradiction_proof=result.contradiction_proof,
            )
        finally:
            conn.close()


def _build_accounting(
    season_id: str,
    start: date,
    end: date,
    batch_no: int,
    feasible: bool,
    conn,
) -> AccountingResponse:
    order_rows = db.load_orders(conn, season_id, batch_no)
    orders = [
        OrderResult(
            order_no=row["order_no"],
            start_date=start + timedelta(days=row["l_idx"] - 1),
            end_date=start + timedelta(days=row["r_idx"] - 1),
            total=row["total"],
        )
        for row in order_rows
    ]

    days: list[DayResult] = []
    official: list[DayResult] = []
    proof: list[str] = []
    if feasible:
        for row in db.load_daily_bounds(conn, season_id, batch_no):
            day = DayResult(
                day_index=row["day_index"],
                date=start + timedelta(days=row["day_index"] - 1),
                min_tickets=row["min_tickets"],
                max_tickets=row["max_tickets"],
                tight=(
                    row["max_tickets"] is not None
                    and row["min_tickets"] == row["max_tickets"]
                ),
            )
            days.append(day)
            if day.tight:
                official.append(day)
    else:
        proof = db.load_contradictions(conn, season_id, batch_no)

    return AccountingResponse(
        season_id=season_id,
        start_date=start,
        end_date=end,
        batch_no=batch_no,
        feasible=feasible,
        contradiction_proof=proof,
        orders=orders,
        days=days,
        official_tickets=official,
    )


@app.get("/api/seasons/{season_id}/accounting", response_model=AccountingResponse)
def get_accounting(season_id: str) -> AccountingResponse:
    conn = db.connect_readonly()
    try:
        season = db.get_season(conn, season_id)
        if season is None:
            raise HTTPException(404, f"演出季 {season_id} 不存在")
        batch_no = db.current_batch_no(conn, season_id)
        if batch_no == 0:
            raise HTTPException(404, f"演出季 {season_id} 尚无生效批次")
        cur = conn.execute(
            "SELECT feasible FROM batches WHERE season_id = ? AND batch_no = ?",
            (season_id, batch_no),
        )
        feasible = bool(cur.fetchone()["feasible"])
        start = date.fromisoformat(season["start_date"])
        end = date.fromisoformat(season["end_date"])
        return _build_accounting(season_id, start, end, batch_no, feasible, conn)
    finally:
        conn.close()


@app.get("/api/seasons", response_model=list[SeasonSummary])
def list_seasons() -> list[SeasonSummary]:
    conn = db.connect_readonly()
    try:
        cur = conn.execute(
            "SELECT s.season_id, s.start_date, s.end_date, b.batch_no, b.feasible "
            "FROM seasons s JOIN batches b "
            "ON b.season_id = s.season_id "
            "AND b.batch_no = (SELECT MAX(batch_no) FROM batches WHERE season_id = s.season_id) "
            "ORDER BY s.season_id"
        )
        return [
            SeasonSummary(
                season_id=row["season_id"],
                start_date=date.fromisoformat(row["start_date"]),
                end_date=date.fromisoformat(row["end_date"]),
                current_batch=row["batch_no"],
                feasible=bool(row["feasible"]),
            )
            for row in cur.fetchall()
        ]
    finally:
        conn.close()


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}
