"""并发回归测试：修复"长事务持写锁"后的交单/查询行为。

覆盖需求：
1. 某季核算期间，其他季能从原入口正常完成交单（不被写锁阻塞）；
2. 同季多笔交单排队时，已生效日报仍可随时查询；
3. 核算或写入失败时沿用旧批次，排队的同季交单仍能继续；
4. 同季同时首交（或同一 seen_batch 的换交）最多一笔生效，其余 409；
5. 写事务内复检到批号已被推进时返回 409，不写入任何数据。
"""

import os
import sqlite3
import tempfile
import threading

import pytest
from fastapi.testclient import TestClient

_TMP = tempfile.mkdtemp(prefix="ticket-api-fix-")
os.environ["TICKET_DB_PATH"] = os.path.join(_TMP, "fix.db")

from app import main, solver  # noqa: E402
from app.main import app  # noqa: E402
from app import database as db  # noqa: E402

START, END = "2026-04-01", "2026-04-05"


@pytest.fixture(scope="module")
def client():
    db.init_db()
    with TestClient(app) as c:
        yield c


def _post(client, season, seen_batch=None, order_no=None, total=1):
    body = {
        "season_id": season,
        "start_date": START,
        "end_date": END,
        "orders": (
            []
            if order_no is None
            else [
                {
                    "order_no": order_no,
                    "start_date": START,
                    "end_date": START,
                    "total": total,
                }
            ]
        ),
    }
    if seen_batch is not None:
        body["seen_batch"] = seen_batch
    return client.post("/api/seasons/submissions", json=body)


def test_other_season_submits_while_one_season_is_computing(client, monkeypatch):
    # 让 S-SLOW 的核算阻塞在一个可控制的同步点上。
    entered = threading.Event()
    release = threading.Event()
    real_solve = solver.solve

    def slow_solve(n_days, orders):
        if orders and any(o.order_no == "SLOW" for o in orders):
            entered.set()
            assert release.wait(timeout=10), "核算线程未在预期时间内进入"
        return real_solve(n_days, orders)

    monkeypatch.setattr(main, "solve", slow_solve)

    # 先给 S-SLOW 建一个生效批次。
    assert _post(client, "S-SLOW").status_code == 200

    errors: list[tuple] = []

    def slow_submit():
        r = _post(client, "S-SLOW", seen_batch=1, order_no="SLOW", total=3)
        if r.status_code != 200:
            errors.append(("slow", r.status_code, r.text))

    t = threading.Thread(target=slow_submit)
    t.start()
    assert entered.wait(timeout=10), "S-SLOW 未进入核算阶段"

    # S-SLOW 正卡在核算里（此时不持有任何 DB 写锁）：
    # 另一个新演出季的首交必须立即成功，不能等到超时报错。
    done = threading.Event()

    def other_submit(season):
        r = _post(client, season, order_no="FAST")
        if r.status_code != 200:
            errors.append((season, r.status_code, r.text))
        done.set()

    ot = threading.Thread(target=other_submit, args=("S-FAST",))
    ot.start()
    assert done.wait(timeout=10), "其他季交单被核算中的季阻塞"
    ot.join()

    release.set()
    t.join(timeout=10)

    assert not errors
    q = client.get("/api/seasons/S-FAST/accounting").json()
    assert q["batch_no"] == 1 and q["feasible"] is True
    q2 = client.get("/api/seasons/S-SLOW/accounting").json()
    assert q2["batch_no"] == 2


def test_accounting_remains_queryable_while_same_season_submissions_queue(
    client, monkeypatch
):
    # 同季先有一笔生效批次。
    assert _post(client, "S-QUEUE", order_no="BASE").status_code == 200

    entered = threading.Event()
    release = threading.Event()
    real_solve = solver.solve

    def blocking_solve(n_days, orders):
        if orders and any(o.order_no == "HOLD" for o in orders):
            entered.set()
            assert release.wait(timeout=10)
        return real_solve(n_days, orders)

    monkeypatch.setattr(main, "solve", blocking_solve)

    # 两笔同季换交同时进入：第一笔卡在核算，第二笔在同季锁上排队。
    def queued(tag):
        _post(client, "S-QUEUE", seen_batch=1, order_no=tag, total=5)

    t1 = threading.Thread(target=queued, args=("HOLD",))
    t2 = threading.Thread(target=queued, args=("WAIT",))
    t1.start()
    assert entered.wait(timeout=10)
    t2.start()

    # 两笔都悬而未决时，已生效日报仍应即时可读，且读到的是旧批。
    for _ in range(5):
        q = client.get("/api/seasons/S-QUEUE/accounting")
        assert q.status_code == 200
        assert q.json()["batch_no"] == 1

    release.set()
    t1.join(timeout=10)
    t2.join(timeout=10)

    # 排队结束：一笔生效（批 2），另一笔因批号过期 409；查询立刻看到新批。
    q = client.get("/api/seasons/S-QUEUE/accounting").json()
    assert q["batch_no"] == 2
    assert len(q["orders"]) == 1


def test_solver_failure_keeps_old_batch_and_queued_submission_proceeds(
    client, monkeypatch
):
    assert _post(client, "S-FAIL", order_no="BASE").status_code == 200

    real_solve = solver.solve
    fail_once = {"on": True}

    def flaky_solve(n_days, orders):
        if fail_once["on"] and orders and any(o.order_no == "BOOM" for o in orders):
            fail_once["on"] = False
            raise RuntimeError("核算引擎内部错误")
        return real_solve(n_days, orders)

    monkeypatch.setattr(main, "solve", flaky_solve)

    # 失败的交单向上抛错（TestClient 下表现为服务端异常），旧批次保持生效。
    with pytest.raises(RuntimeError):
        _post(client, "S-FAIL", seen_batch=1, order_no="BOOM")

    q = client.get("/api/seasons/S-FAIL/accounting").json()
    assert q["batch_no"] == 1  # 沿用生效批次

    # 同季锁已释放，后续交单可继续，凭批号 1 正常换交成批 2。
    r = _post(client, "S-FAIL", seen_batch=1, order_no="RECOVER", total=7)
    assert r.status_code == 200 and r.json()["batch_no"] == 2
    q2 = client.get("/api/seasons/S-FAIL/accounting").json()
    assert q2["batch_no"] == 2


def test_write_failure_rolls_back_and_keeps_effective_batch(client, monkeypatch):
    assert _post(client, "S-WRITE", order_no="BASE").status_code == 200

    def broken_insert(*args, **kwargs):
        raise sqlite3.IntegrityError("UNIQUE constraint failed: batches.batch_no")

    # 模拟写入阶段失败：insert_batch 抛 IntegrityError，应转为 409 并回滚。
    monkeypatch.setattr(db, "insert_batch", broken_insert)
    r = _post(client, "S-WRITE", seen_batch=1, order_no="X")
    assert r.status_code == 409
    monkeypatch.undo()

    # 旧批不受影响，且同季后续交单仍能继续。
    q = client.get("/api/seasons/S-WRITE/accounting").json()
    assert q["batch_no"] == 1
    r2 = _post(client, "S-WRITE", seen_batch=1, order_no="Y", total=2)
    assert r2.status_code == 200 and r2.json()["batch_no"] == 2


def test_concurrent_first_submissions_only_one_takes_effect(client):
    season = "S-FIRST"
    results: list[int] = []
    barrier = threading.Barrier(6)

    def worker(i):
        barrier.wait()
        r = _post(client, season, order_no=f"F{i}", total=i)
        results.append(r.status_code)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert results.count(200) == 1
    assert results.count(409) == 5
    q = client.get(f"/api/seasons/{season}/accounting").json()
    assert q["batch_no"] == 1
    assert len(q["orders"]) == 1


def test_stale_batch_detected_inside_write_transaction(client, monkeypatch):
    # 同季首交成功，批号 1。
    assert _post(client, "S-EXT", order_no="BASE").status_code == 200

    real_current = db.current_batch_no
    calls = {"n": 0}

    def current_with_external_advance(conn, season_id):
        # 该季第二次读取批号（即写事务内复检）时，模拟批号已被进程外写入者推进。
        if season_id == "S-EXT":
            calls["n"] += 1
            if calls["n"] == 2:
                return 2
        return real_current(conn, season_id)

    monkeypatch.setattr(db, "current_batch_no", current_with_external_advance)
    r = _post(client, "S-EXT", seen_batch=1, order_no="LATE")
    assert r.status_code == 409
    monkeypatch.undo()

    # 真实批号仍是 1（本请求未写入任何数据，外部推进仅为桩值）。
    q = client.get("/api/seasons/S-EXT/accounting").json()
    assert q["batch_no"] == 1
    assert {o["order_no"] for o in q["orders"]} == {"BASE"}
