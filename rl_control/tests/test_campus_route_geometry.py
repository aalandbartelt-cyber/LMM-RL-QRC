import unittest
from pathlib import Path

from go2_foundation.route_mission import MissionWaypoint, load_route_file
from go2_foundation.route_validation import AxisAlignedObstacle, validate_route_clearance


ROOT = Path(__file__).parents[2]
ROUTE = ROOT / "simulation" / "scenes" / "campus_security" / "physical_route.json"
BUILDINGS = (
    AxisAlignedObstacle("office", -23.0, -5.0, 7.0, 17.0),
    AxisAlignedObstacle("warehouse", 3.5, 22.5, 7.0, 17.0),
    AxisAlignedObstacle("power_room", -21.0, -13.0, -5.0, 1.0),
    AxisAlignedObstacle("gate_house", -9.5, -4.5, -18.95, -15.45),
)


class CampusRouteGeometryTests(unittest.TestCase):
    def test_original_preview_segment_crosses_office(self):
        route = (
            MissionWaypoint("P2", -24.0, 5.0),
            MissionWaypoint("P3", -7.0, 15.0),
        )
        issues = validate_route_clearance(route, BUILDINGS, clearance_m=0.45)
        self.assertEqual(issues[0].obstacle_name, "office")

    def test_physical_route_clears_buildings(self):
        route = load_route_file(ROUTE)
        self.assertEqual(validate_route_clearance(route, BUILDINGS, clearance_m=0.45), ())
        self.assertEqual(route[0].task, "mission_start")
        self.assertEqual(route[-1].task, "mission_finish")

    def test_power_room_checkpoint_matches_scene_geometry(self):
        route = load_route_file(ROUTE)
        checkpoint = next(point for point in route if point.task == "equipment_temperature")
        self.assertLess(checkpoint.x_m, -21.0)
        self.assertGreaterEqual(checkpoint.y_m, -5.0)
        self.assertLessEqual(checkpoint.y_m, 1.0)


if __name__ == "__main__":
    unittest.main()
