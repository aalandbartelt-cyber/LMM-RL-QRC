from simulation.demo_media.scenarios import build_scenarios
import numpy as np

from simulation.demo_media.rendering import interpolate_route, render_frame
from simulation.demo_media.mujoco_route import build_scene_xml, compact_route, route_command
from simulation.demo_media.generate_submission_media import sha256_file, validate_video


def test_two_submission_scenarios_have_complete_missions():
    scenarios = build_scenarios()

    assert set(scenarios) == {"campus_security", "disaster_response"}
    assert len(scenarios["campus_security"].route) == 19
    assert scenarios["campus_security"].events[-1].status == "任务完成"
    assert scenarios["disaster_response"].events[-1].status == "安全返航"


def test_evidence_labels_do_not_claim_real_robot():
    for scenario in build_scenarios().values():
        assert "仿真" in scenario.evidence_label
        assert "真机" not in scenario.evidence_label


def test_interpolate_route_starts_and_ends_at_route_bounds():
    scenario = build_scenarios()["campus_security"]

    assert interpolate_route(scenario.route, 0.0)[:2] == (
        scenario.route[0].x,
        scenario.route[0].y,
    )
    assert interpolate_route(scenario.route, 1.0)[:2] == (
        scenario.route[-1].x,
        scenario.route[-1].y,
    )


def test_rendered_frame_is_full_hd_rgb():
    frame = render_frame(build_scenarios()["campus_security"], 0.5, None)

    assert isinstance(frame, np.ndarray)
    assert frame.shape == (1080, 1920, 3)
    assert frame.dtype == np.uint8


def test_compact_route_starts_at_origin_and_fits_local_scene():
    route = compact_route(build_scenarios()["campus_security"])

    assert (route[0].x, route[0].y) == (0.0, 0.0)
    assert max(abs(point.x) for point in route) <= 4.0
    assert max(abs(point.y) for point in route) <= 4.0


def test_route_command_moves_forward_when_aligned():
    route = compact_route(build_scenarios()["campus_security"])
    yaw = np.arctan2(route[1].y - route[0].y, route[1].x - route[0].x)

    command = route_command(0.0, 0.0, float(yaw), route[1], 0.5)

    assert command.forward_mps > 0.0
    assert abs(command.yaw_rate_rps) <= 1.0
    assert not command.reached


def test_route_command_turns_in_place_when_facing_away():
    target = build_scenarios()["disaster_response"].route[1]

    command = route_command(target.x - 1.0, target.y, 3.14159, target, 0.35)

    assert command.forward_mps == 0.0
    assert abs(command.yaw_rate_rps) > 0.5


def test_route_command_accepts_safe_waypoint_tolerance():
    target = build_scenarios()["campus_security"].route[1]

    command = route_command(target.x - 0.35, target.y, 0.0, target, 0.5)

    assert command.reached


def test_scene_xml_uses_explicit_go2_mesh_directory(tmp_path):
    go2_xml = tmp_path / "go2.xml"
    go2_xml.write_text("<mujoco/>", encoding="utf-8")

    xml = build_scene_xml(build_scenarios()["campus_security"], go2_xml)

    expected_assets = (tmp_path / "assets").resolve().as_posix()
    assert f'meshdir="{expected_assets}"' in xml


def test_sha256_file_is_stable(tmp_path):
    path = tmp_path / "sample.bin"
    path.write_bytes(b"go2")

    assert sha256_file(path) == "be4fb8841ddfa79756046f79946b71a6147cb1c1d2ed13cd3430d04a197eafc3"


def test_validate_video_accepts_written_clip(tmp_path):
    import cv2

    path = tmp_path / "clip.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10, (320, 180))
    for _ in range(12):
        writer.write(np.zeros((180, 320, 3), dtype=np.uint8))
    writer.release()

    assert validate_video(path, min_frames=10, width=320, height=180, fps=10)
