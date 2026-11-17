"""取消受理与构建终态竞态的专项测试。

覆盖需求中的关键交错：
1. 提交后立即取消（构建任务可能尚未首跑）：仍必须受理，且没有任何步骤启动。
2. 取消与终态收尾交错：数据库守卫把两个动作线性化，终态先写入则取消被忽略、
   原记录（status / cancel_requested / finished_at）一字不改。
3. 受理后未启动/依赖步骤不得启动；运行中命令及其后代清空后才确认取消、释放名额。
4. API 重启后不补跑已取消构建的任何步骤。
"""
from __future__ import annotations

import asyncio
import sqlite3
import time

import pytest

from test_integration import (
    ServerHarness,
    _free_port,
    assert_pid_dead,
    step,
    submit,
    wait_build,
)

TERMINAL = {"succeeded", "failed", "cancelled", "interrupted"}


def step_map(detail: dict) -> dict:
    return {s["id"]: s for s in detail["steps"]}


@pytest.fixture(scope="module")
def server(tmp_path_factory):
    srv = ServerHarness(tmp_path_factory.mktemp("race-data"), _free_port())
    srv.start()
    yield srv
    srv.stop()


# --------------------------------------- 1. 提交后立即取消：受理且零步骤执行

def test_cancel_immediately_after_submit_is_accepted_and_no_step_runs(server):
    # 不等待任何步骤进入 running 就取消，专门打击"事件尚未登记/任务未首跑"空窗。
    # 修复前该取消会丢失，构建最终 succeeded；修复后必须受理并以 cancelled 收尾。
    bid = submit(server.client, "immediate-cancel", [
        step("a", "echo a-ran; sleep 20"),
        step("b", "echo b-ran; sleep 20"),
        step("c", "echo c-ran", ["a"]),
    ])
    r = server.client.post(f"/builds/{bid}/cancel")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["cancel_requested"] is True
    assert "受理" in body["detail"]

    detail = wait_build(server.client, bid, timeout=20)
    # 关键断言：取消不丢，构建绝不会以 succeeded 收尾
    assert detail["status"] == "cancelled"
    sm = step_map(detail)
    # 依赖步骤 c 没有任何启动机会（受理后未启动步骤一律不得执行）
    assert sm["c"]["status"] == "cancelled"
    log_c = server.client.get(f"/builds/{bid}/steps/c/logs").json()
    cmd_out_c = "\n".join(
        line for line in log_c["content"].splitlines()
        if not line.startswith("[runner]")
    )
    assert "c-ran" not in cmd_out_c
    # 没有任何步骤可以成功
    assert not any(s["status"] == "succeeded" for s in detail["steps"])


# ----------------------------- 2. 终态先写入：取消明确忽略，记录保持不变

def test_cancel_after_terminal_is_ignored_and_record_untouched(server):
    bid = submit(server.client, "too-late-cancel", [step("a", "echo ok")])
    detail = wait_build(server.client, bid)
    assert detail["status"] == "succeeded"
    assert detail["cancel_requested"] is False
    finished_at = detail["finished_at"]

    r = server.client.post(f"/builds/{bid}/cancel")
    assert r.status_code == 200
    body = r.json()
    assert "终态" in body["detail"]
    assert body["status"] == "succeeded"
    assert body["cancel_requested"] is False  # 守卫未命中，标志位没有被置上

    fresh = server.client.get(f"/builds/{bid}").json()
    assert fresh["status"] == "succeeded"
    assert fresh["cancel_requested"] is False
    assert fresh["finished_at"] == finished_at  # 原记录未被改写

    # 再来一次仍然幂等忽略
    r2 = server.client.post(f"/builds/{bid}/cancel")
    assert r2.status_code == 200 and "终态" in r2.json()["detail"]


# ---------------- 2b. 守卫线性化的确定性验证（直接驱动数据库，两种顺序各测一次）

def test_db_guard_linearizes_cancel_and_terminal_in_both_orders(tmp_path):
    from app.config import ensure_dirs
    from app.db import Database

    async def scenario():
        import app.config as config
        config.DATA_DIR = tmp_path
        config.DB_PATH = tmp_path / "p.db"
        config.WORKDIR_ROOT = tmp_path / "workdir"
        config.LOG_ROOT = tmp_path / "logs"
        ensure_dirs()

        db = Database()
        await db.connect()
        try:
            # 顺序 A：取消先受理 -> 终态后写入；受理返回 True
            id_a = await db.create_build("A", "2026-01-01T00:00:00+00:00")
            assert await db.request_cancel_if_active(id_a) is True
            row_a = await db.get_build(id_a)
            assert row_a["cancel_requested"] == 1
            # 调度器收尾写入终态（只允许一次）
            assert await db.finish_build_if_active(
                id_a, "cancelled", "2026-01-01T00:00:01+00:00", "构建被取消"
            ) is True
            # 终态之后再到的取消必须被忽略
            assert await db.request_cancel_if_active(id_a) is False
            row_a2 = await db.get_build(id_a)
            assert row_a2["status"] == "cancelled"
            assert row_a2["cancel_requested"] == 1

            # 顺序 B：终态先写入 -> 取消被忽略，cancel_requested 保持 0
            id_b = await db.create_build("B", "2026-01-01T00:00:00+00:00")
            assert await db.finish_build_if_active(
                id_b, "succeeded", "2026-01-01T00:00:01+00:00", "全部步骤成功"
            ) is True
            assert await db.request_cancel_if_active(id_b) is False
            row_b = await db.get_build(id_b)
            assert row_b["status"] == "succeeded"
            assert row_b["cancel_requested"] == 0

            # 终态写两次：第二次守卫拒绝，记录保持第一次的内容
            assert await db.finish_build_if_active(
                id_b, "cancelled", "2026-01-01T00:00:02+00:00", "不应生效"
            ) is False
            row_b2 = await db.get_build(id_b)
            assert row_b2["status"] == "succeeded"
            assert row_b2["detail"] == "全部步骤成功"
            assert row_b2["finished_at"] == "2026-01-01T00:00:01+00:00"
        finally:
            await db.close()

    asyncio.run(scenario())


# ------------- 3. 受理后名额只在进程组（含后代）清空后释放，新构建可占满两名额

def test_cancel_clears_descendants_and_releases_both_slots(server):
    # 顽固后代忽略 SIGTERM，验证取消路径同样要走完 SIGKILL 排空才确认/释放
    def stubborn(uid: str) -> str:
        return (
            f"trap '' TERM; sleep 300 & echo $! > child-{uid}.pid; "
            "echo main-start; wait"
        )

    bid = submit(server.client, "cancel-slot", [
        step("x", stubborn("x"), timeout=30),
        step("y", stubborn("y"), timeout=30),
    ])
    workdir = server.data_dir / "workdir" / str(bid)
    deadline = time.time() + 10
    while time.time() < deadline:
        sm = step_map(server.client.get(f"/builds/{bid}").json())
        if all(v["status"] == "running" for v in sm.values()) and \
                (workdir / "child-x.pid").exists() and \
                (workdir / "child-y.pid").exists():
            break
        time.sleep(0.1)
    else:
        raise AssertionError("两个步骤未同时进入 running 或后代 PID 未落盘")

    r = server.client.post(f"/builds/{bid}/cancel")
    assert r.status_code == 200 and r.json()["cancel_requested"] is True
    detail = wait_build(server.client, bid, timeout=30)
    assert detail["status"] == "cancelled"

    # 终态已确认意味着进程组已清空：两个步骤各自的顽固后代必须已死
    child_pids = [
        int(pid_file.read_text().strip())
        for pid_file in workdir.glob("child-*.pid")
    ]
    assert len(child_pids) == 2
    for pid in child_pids:
        assert_pid_dead(pid)

    # 名额已全部释放：新构建的两个独立步骤可以并行跑满 2 个名额
    t0 = time.time()
    new_bid = submit(server.client, "after-cancel-slots", [
        step("p1", "echo p1; sleep 1.0"),
        step("p2", "echo p2; sleep 1.0"),
    ])
    new_detail = wait_build(server.client, new_bid, timeout=15)
    elapsed = time.time() - t0
    assert new_detail["status"] == "succeeded"
    assert elapsed < 2.5, f"名额疑似未释放，两个并行步骤耗时 {elapsed:.2f}s"


# ---------------------- 4. 重启后不补跑已取消构建：未启动步骤永远不执行

def test_restart_never_reruns_cancelled_build(tmp_path_factory):
    data_dir = tmp_path_factory.mktemp("cancel-restart-data")
    port = _free_port()
    srv = ServerHarness(data_dir, port)
    srv.start()
    try:
        bid = submit(srv.client, "cancelled-then-restart", [
            step("a", "echo a-start; sleep 30; echo a-end", timeout=60),
            step("b", "echo b-never-run", ["a"], timeout=60),
            # 独立的长任务：取消受理后应被终止，重启后绝不能继续/补跑
            step("c", "echo c-start; sleep 30; echo c-never-finish", timeout=60),
        ])
        # 立即取消并马上强杀 API，刻意制造"受理后清理未完成"的交错
        r = srv.client.post(f"/builds/{bid}/cancel")
        assert r.status_code == 200 and r.json()["cancel_requested"] is True
        srv.stop()  # SIGTERM 后等待退出；步骤进程组可能仍在清理窗口内
    finally:
        srv.client.close()

    srv2 = ServerHarness(data_dir, port)
    srv2.start()
    try:
        detail = srv2.client.get(f"/builds/{bid}").json()
        # 取消已受理：终态要么已是 cancelled，要么被恢复标为 interrupted，
        # 但任何步骤都不允许在重启后被补跑
        assert detail["status"] in TERMINAL
        sm = step_map(detail)
        for uid, st in sm.items():
            assert st != "running"
        # b 从未满足依赖、c 的尾部输出不可能产生：重启后没有任何补跑
        assert sm["b"]["status"] in ("cancelled", "interrupted")
        assert sm["c"]["status"] in ("cancelled", "interrupted")

        for uid in ("b", "c"):
            log = srv2.client.get(f"/builds/{bid}/steps/{uid}/logs").json()
            cmd_out = "\n".join(
                line for line in log["content"].splitlines()
                if not line.startswith("[runner]")
            )
            assert f"{uid}-never" not in cmd_out
        # b 是依赖步骤，从头到尾不该产生任何命令输出
        log_b = srv2.client.get(f"/builds/{bid}/steps/b/logs").json()
        assert "b-never-run" not in log_b["content"]

        # DB 侧终态守恒：不存在 pending/running/cleaning_up 的残留步骤
        conn = sqlite3.connect(data_dir / "pipeline.db")
        try:
            states = {
                row[0]: row[1]
                for row in conn.execute(
                    "SELECT step_uid, status FROM steps WHERE build_id=?", (bid,)
                )
            }
        finally:
            conn.close()
        assert set(states.values()) <= {"succeeded", "failed", "skipped",
                                        "cancelled", "interrupted"}

        # 恢复后新构建照常工作
        new_id = submit(srv2.client, "after-cancel-restart", [step("z", "echo alive")])
        assert wait_build(srv2.client, new_id)["status"] == "succeeded"
    finally:
        srv2.stop()
