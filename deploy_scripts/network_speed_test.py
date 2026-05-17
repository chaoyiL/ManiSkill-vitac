from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import statistics
import sys
import threading
import time
from typing import Any

import numpy as np
from websockets.sync.server import ServerConnection
from websockets.sync.server import serve

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))
os.chdir(ROOT_DIR)

from client.interface_client import InterfaceClient  # noqa: E402
from client.interface_client import _Packer  # noqa: E402
from client.interface_client import _unpackb  # noqa: E402


IMAGE_KEYS = (
    "observation.images.camera0",
    "observation.images.camera1",
    "observation.images.tactile_left_0",
    "observation.images.tactile_right_0",
    "observation.images.tactile_left_1",
    "observation.images.tactile_right_1",
)


def now_ns() -> tuple[int, int]:
    return time.time_ns(), time.monotonic_ns()


def make_random_observation(
    rng: np.random.Generator,
    image_shape: tuple[int, int, int],
    state_dim: int,
) -> dict[str, Any]:
    obs = {
        key: rng.integers(0, 256, size=image_shape, dtype=np.uint8)
        for key in IMAGE_KEYS
    }
    obs["observation.state"] = rng.random(state_dim, dtype=np.float32)
    obs["task"] = "network speed test"
    return obs


def make_random_action(
    rng: np.random.Generator,
    action_horizon: int,
    action_dim: int,
) -> np.ndarray:
    return rng.random((action_horizon, action_dim), dtype=np.float32)


def pack_message(message: dict[str, Any]) -> bytes:
    return _Packer().pack(message)


def send_packed(ws: ServerConnection, message: dict[str, Any]) -> int:
    packed = pack_message(message)
    ws.send(packed)
    return len(packed)


def recv_message(ws: ServerConnection, timeout: float | None = None) -> dict[str, Any]:
    raw_message = ws.recv(timeout=timeout)
    if isinstance(raw_message, str):
        raise RuntimeError("网络测速脚本只接收二进制 websocket 帧。")
    return _unpackb(raw_message)


def write_jsonl(path: Path | None, row: dict[str, Any]) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def summarize(values_ms: list[float]) -> dict[str, float] | None:
    if not values_ms:
        return None
    return {
        "count": len(values_ms),
        "mean_ms": statistics.fmean(values_ms),
        "min_ms": min(values_ms),
        "max_ms": max(values_ms),
        "p50_ms": statistics.median(values_ms),
        "p95_ms": statistics.quantiles(values_ms, n=20)[18] if len(values_ms) >= 2 else values_ms[0],
    }


def print_summary(name: str, values_ms: list[float]) -> None:
    summary = summarize(values_ms)
    if summary is None:
        print(f"[{name}] 无样本")
        return
    print(
        f"[{name}] 样本数={summary['count']} 平均={summary['mean_ms']:.3f} ms "
        f"p50={summary['p50_ms']:.3f} ms p95={summary['p95_ms']:.3f} ms "
        f"最小={summary['min_ms']:.3f} ms 最大={summary['max_ms']:.3f} ms"
    )


def format_ms(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.3f} ms"


def estimate_clock_offset_robot(args: argparse.Namespace, ws: ServerConnection) -> dict[str, Any]:
    """Estimate inference_wall_ns - robot_wall_ns with an NTP-style exchange."""
    samples = []
    for sample_idx in range(args.clock_sync_samples):
        robot_send_wall_ns, _ = now_ns()
        send_packed(
            ws,
            {
                "type": "clock_sync_request",
                "sample_idx": sample_idx,
                "robot_send_wall_ns": robot_send_wall_ns,
            },
        )
        response = recv_message(ws, timeout=args.timeout)
        robot_recv_wall_ns, _ = now_ns()
        if response.get("type") != "clock_sync_response":
            raise RuntimeError(f"校准时钟时收到非预期消息: {response}")

        infer_recv_wall_ns = int(response["infer_recv_wall_ns"])
        infer_send_wall_ns = int(response["infer_send_wall_ns"])
        rtt_ns = (robot_recv_wall_ns - robot_send_wall_ns) - (infer_send_wall_ns - infer_recv_wall_ns)
        offset_ns = (
            (infer_recv_wall_ns - robot_send_wall_ns)
            + (infer_send_wall_ns - robot_recv_wall_ns)
        ) / 2.0
        samples.append(
            {
                "sample_idx": sample_idx,
                "offset_ns": offset_ns,
                "rtt_ns": rtt_ns,
                "robot_send_wall_ns": robot_send_wall_ns,
                "robot_recv_wall_ns": robot_recv_wall_ns,
                "infer_recv_wall_ns": infer_recv_wall_ns,
                "infer_send_wall_ns": infer_send_wall_ns,
            }
        )

    best_sample = min(samples, key=lambda item: item["rtt_ns"])
    sync_info = {
        "clock_offset_ns": best_sample["offset_ns"],
        "clock_offset_ms": best_sample["offset_ns"] / 1e6,
        "clock_sync_rtt_ms": best_sample["rtt_ns"] / 1e6,
        "clock_sync_samples": samples,
    }
    send_packed(ws, {"type": "clock_sync_done", **sync_info})
    print(
        f"[机器人端] 时钟偏移=推理端-机器人端={sync_info['clock_offset_ms']:.3f} ms "
        f"(最佳校准RTT={sync_info['clock_sync_rtt_ms']:.3f} ms)"
    )
    return sync_info


def handle_clock_sync_inference(client: InterfaceClient, timeout: float) -> dict[str, Any]:
    while True:
        message = client._recv_message(timeout=timeout)
        recv_wall_ns, _ = now_ns()
        message_type = message.get("type")
        if message_type == "clock_sync_request":
            response = {
                "type": "clock_sync_response",
                "sample_idx": int(message["sample_idx"]),
                "robot_send_wall_ns": int(message["robot_send_wall_ns"]),
                "infer_recv_wall_ns": recv_wall_ns,
            }
            send_wall_ns, _ = now_ns()
            response["infer_send_wall_ns"] = send_wall_ns
            client._ws.send(pack_message(response))
            continue
        if message_type == "clock_sync_done":
            print(
                f"[推理端] 时钟偏移=推理端-机器人端={message['clock_offset_ms']:.3f} ms "
                f"(最佳校准RTT={message['clock_sync_rtt_ms']:.3f} ms)"
            )
            return message
        raise RuntimeError(f"校准时钟时收到非预期消息: {message}")


def robot_handler(args: argparse.Namespace, stop_event: threading.Event, ws: ServerConnection) -> None:
    rng = np.random.default_rng(args.seed)
    send_packed(ws, {"type": "hello", "role": "network_speed_robot"})
    sync_info = estimate_clock_offset_robot(args, ws)
    clock_offset_ns = float(sync_info["clock_offset_ns"])

    rtt_ms: list[float] = []
    obs_send_ms: list[float] = []
    action_wait_ms: list[float] = []
    action_send_to_recv_ms: list[float] = []

    for obs_seq in range(args.iterations):
        obs = make_random_observation(rng, args.image_shape, args.state_dim)
        send_wall_ns, send_mono_ns = now_ns()
        obs_message = {
            "type": "obs",
            "obs_seq": obs_seq,
            "obs": obs,
            "robot_timestamps": {
                "obs_send_wall_ns": send_wall_ns,
                "obs_send_mono_ns": send_mono_ns,
            },
        }

        packed_obs = pack_message(obs_message)
        ws.send(packed_obs)
        _, send_done_mono_ns = now_ns()

        action_message = recv_message(ws, timeout=args.timeout)
        recv_wall_ns, recv_mono_ns = now_ns()
        if action_message.get("type") != "action":
            raise RuntimeError(f"等待 action 时收到非预期消息: {action_message}")
        if int(action_message["obs_seq"]) != obs_seq:
            raise RuntimeError(f"Action 的 obs_seq 不匹配: 期望 {obs_seq}, 实际 {action_message['obs_seq']}")

        action = np.asarray(action_message["action"])
        if action.shape != (args.action_horizon, args.action_dim):
            raise RuntimeError(f"Action 形状不匹配: 期望 {(args.action_horizon, args.action_dim)}, 实际 {action.shape}")

        current_rtt_ms = (recv_mono_ns - send_mono_ns) / 1e6
        current_obs_send_ms = (send_done_mono_ns - send_mono_ns) / 1e6
        infer_timestamps = action_message.get("infer_timestamps", {})
        infer_send_mono_ns = infer_timestamps.get("action_send_start_mono_ns")
        if isinstance(infer_send_mono_ns, int):
            current_action_wait_ms = (recv_mono_ns - infer_send_mono_ns) / 1e6
            action_wait_ms.append(current_action_wait_ms)
        else:
            current_action_wait_ms = None
        infer_send_wall_ns = infer_timestamps.get("action_send_start_wall_ns")
        if isinstance(infer_send_wall_ns, int):
            current_action_send_to_recv_ms = (recv_wall_ns - (infer_send_wall_ns - clock_offset_ns)) / 1e6
            action_send_to_recv_ms.append(current_action_send_to_recv_ms)
        else:
            current_action_send_to_recv_ms = None

        rtt_ms.append(current_rtt_ms)
        obs_send_ms.append(current_obs_send_ms)

        row = {
            "role": "robot",
            "obs_seq": obs_seq,
            "obs_payload_bytes": len(packed_obs),
            "action_payload_bytes": len(pack_message(action_message)),
            "robot_obs_send_wall_ns": send_wall_ns,
            "robot_obs_send_mono_ns": send_mono_ns,
            "robot_obs_send_done_mono_ns": send_done_mono_ns,
            "robot_action_recv_wall_ns": recv_wall_ns,
            "robot_action_recv_mono_ns": recv_mono_ns,
            "rtt_ms": current_rtt_ms,
            "obs_send_call_ms": current_obs_send_ms,
            "action_wait_after_infer_send_ms": current_action_wait_ms,
            "action_send_to_recv_wall_ms": current_action_send_to_recv_ms,
            "clock_offset_ns": clock_offset_ns,
            "clock_sync_rtt_ms": sync_info["clock_sync_rtt_ms"],
            "infer_timestamps": infer_timestamps,
        }
        write_jsonl(args.jsonl, row)

        if obs_seq % args.log_every == 0 or obs_seq == args.iterations - 1:
            print(
                f"[机器人端] 序号={obs_seq} 往返耗时={current_rtt_ms:.3f} ms "
                f"action发送到接收={format_ms(current_action_send_to_recv_ms)} "
                f"obs发送调用耗时={current_obs_send_ms:.3f} ms obs字节数={len(packed_obs)}"
            )

        if args.rate_hz > 0 and obs_seq < args.iterations - 1:
            time.sleep(1.0 / args.rate_hz)

    print_summary("机器人端 往返耗时", rtt_ms)
    print_summary("机器人端 obs发送调用耗时", obs_send_ms)
    print_summary("机器人端 action发送后等待耗时", action_wait_ms)
    print_summary("机器人端 action发送到接收耗时", action_send_to_recv_ms)
    stop_event.set()


def run_robot(args: argparse.Namespace) -> None:
    stop_event = threading.Event()

    def handler(ws: ServerConnection) -> None:
        robot_handler(args, stop_event, ws)

    with serve(handler, args.host, args.port, compression=None, max_size=None) as server:
        print(f"[机器人端] 正在监听 ws://{args.host}:{args.port}")
        server_thread = threading.Thread(target=server.serve_forever, daemon=True)
        server_thread.start()
        try:
            stop_event.wait()
        except KeyboardInterrupt:
            print("[机器人端] 已中断")
        finally:
            server.shutdown()
            server_thread.join(timeout=2.0)


def run_inference(args: argparse.Namespace) -> None:
    rng = np.random.default_rng(args.seed)
    client = InterfaceClient(args.ip, args.port, token=args.token, add_port=args.add_port)
    sync_info = handle_clock_sync_inference(client, args.timeout)
    clock_offset_ns = float(sync_info["clock_offset_ns"])
    recv_to_send_ms: list[float] = []
    send_call_ms: list[float] = []
    obs_send_to_recv_ms: list[float] = []

    try:
        for _ in range(args.iterations):
            message = client._recv_message(timeout=args.timeout)
            recv_wall_ns, recv_mono_ns = now_ns()
            if message.get("type") != "obs":
                raise RuntimeError(f"等待 obs 时收到非预期消息: {message}")

            obs_seq = int(message["obs_seq"])
            robot_timestamps = message.get("robot_timestamps", {})
            robot_obs_send_wall_ns = robot_timestamps.get("obs_send_wall_ns")
            if isinstance(robot_obs_send_wall_ns, int):
                current_obs_send_to_recv_ms = (recv_wall_ns - (robot_obs_send_wall_ns + clock_offset_ns)) / 1e6
                obs_send_to_recv_ms.append(current_obs_send_to_recv_ms)
            else:
                current_obs_send_to_recv_ms = None
            action = make_random_action(rng, args.action_horizon, args.action_dim)
            send_wall_ns, send_mono_ns = now_ns()
            action_message = {
                "type": "action",
                "obs_seq": obs_seq,
                "action": action,
                "infer_timestamps": {
                    "obs_recv_wall_ns": recv_wall_ns,
                    "obs_recv_mono_ns": recv_mono_ns,
                    "action_send_start_wall_ns": send_wall_ns,
                    "action_send_start_mono_ns": send_mono_ns,
                },
            }
            packed_action = pack_message(action_message)
            client._ws.send(packed_action)
            _, send_done_mono_ns = now_ns()

            current_recv_to_send_ms = (send_mono_ns - recv_mono_ns) / 1e6
            current_send_call_ms = (send_done_mono_ns - send_mono_ns) / 1e6
            recv_to_send_ms.append(current_recv_to_send_ms)
            send_call_ms.append(current_send_call_ms)

            row = {
                "role": "inference",
                "obs_seq": obs_seq,
                "obs_payload_bytes": len(pack_message(message)),
                "action_payload_bytes": len(packed_action),
                "robot_timestamps": robot_timestamps,
                "infer_obs_recv_wall_ns": recv_wall_ns,
                "infer_obs_recv_mono_ns": recv_mono_ns,
                "infer_action_send_start_wall_ns": send_wall_ns,
                "infer_action_send_start_mono_ns": send_mono_ns,
                "infer_action_send_done_mono_ns": send_done_mono_ns,
                "obs_send_to_recv_wall_ms": current_obs_send_to_recv_ms,
                "clock_offset_ns": clock_offset_ns,
                "clock_sync_rtt_ms": sync_info["clock_sync_rtt_ms"],
                "recv_to_send_ms": current_recv_to_send_ms,
                "action_send_call_ms": current_send_call_ms,
            }
            write_jsonl(args.jsonl, row)

            if obs_seq % args.log_every == 0 or obs_seq == args.iterations - 1:
                print(
                    f"[推理端] 序号={obs_seq} 接收后到发送={current_recv_to_send_ms:.3f} ms "
                    f"obs发送到接收={format_ms(current_obs_send_to_recv_ms)} "
                    f"action发送调用耗时={current_send_call_ms:.3f} ms action字节数={len(packed_action)}"
                )
    finally:
        client.close()

    print_summary("推理端 接收后到发送耗时", recv_to_send_ms)
    print_summary("推理端 action发送调用耗时", send_call_ms)
    print_summary("推理端 obs发送到接收耗时", obs_send_to_recv_ms)


def parse_image_shape(value: str) -> tuple[int, int, int]:
    parts = tuple(int(part) for part in value.lower().replace("x", ",").split(","))
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("图像形状必须是 H,W,C，例如 224,224,3")
    return parts


def add_common_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--iterations", "-n", type=int, default=100, help="obs/action 交换次数。")
    parser.add_argument("--image-shape", type=parse_image_shape, default=(224, 224, 3), help="Observation 图像形状 H,W,C。")
    parser.add_argument("--state-dim", type=int, default=20, help="Observation state 维度。")
    parser.add_argument("--action-horizon", type=int, default=15, help="Action horizon。")
    parser.add_argument("--action-dim", type=int, default=20, help="Action 维度。")
    parser.add_argument("--timeout", type=float, default=30.0, help="接收超时时间，单位秒。")
    parser.add_argument("--clock-sync-samples", type=int, default=10, help="NTP 风格时钟偏移采样次数。")
    parser.add_argument("--seed", type=int, default=0, help="随机种子。")
    parser.add_argument("--log-every", type=int, default=10, help="每 N 个样本打印一次日志。")
    parser.add_argument("--jsonl", type=Path, default=None, help="可选 JSONL 文件，用于保存每个样本的时间戳。")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="测量 websocket obs/action 传输延迟。")
    subparsers = parser.add_subparsers(dest="role", required=True)

    robot_parser = subparsers.add_parser("robot", help="运行机器人端 websocket server。")
    add_common_options(robot_parser)
    robot_parser.add_argument("--host", default="0.0.0.0", help="监听地址。")
    robot_parser.add_argument("--port", type=int, default=8000, help="监听端口。")
    robot_parser.add_argument("--rate-hz", type=float, default=0.0, help="可选发送频率限制；0 表示不限制。")
    robot_parser.set_defaults(func=run_robot)

    inference_parser = subparsers.add_parser("inference", help="运行推理端 websocket client。")
    add_common_options(inference_parser)
    inference_parser.add_argument("--ip", default="127.0.0.1", help="机器人端 websocket 地址或 URL。")
    inference_parser.add_argument("--port", default="8000", help="机器人端 websocket 端口。")
    inference_parser.add_argument("--add-port", default=None, action=argparse.BooleanOptionalAction)
    inference_parser.add_argument("--token", default=None, help="可选 bearer token，与 InterfaceClient 一致。")
    inference_parser.set_defaults(func=run_inference)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.iterations <= 0:
        parser.error("--iterations 必须为正数")
    if args.log_every <= 0:
        parser.error("--log-every 必须为正数")
    if args.clock_sync_samples <= 0:
        parser.error("--clock-sync-samples 必须为正数")
    args.func(args)


if __name__ == "__main__":
    main()
