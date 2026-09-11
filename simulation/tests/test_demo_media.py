from simulation.demo_media.scenarios import build_scenarios


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
