#include <gtest/gtest.h>

#include <opencv2/opencv.hpp>

#include "polygon_processor.h"
#include "proto/exchange_protocol.pb.h"

namespace {

// Helper function to create a polygon config
exchange_protocol::PolygonConfig CreatePolygonConfig(
    const std::vector<std::pair<int, int>>& points,
    exchange_protocol::PolygonType type, int priority,
    const std::vector<std::string>& class_filters) {
  exchange_protocol::PolygonConfig config;

  for (const auto& p : points) {
    auto* point = config.add_points();
    point->set_x(p.first);
    point->set_y(p.second);
  }

  config.set_type(type);
  config.set_priority(priority);

  for (const auto& filter : class_filters) {
    config.add_class_filters(filter);
  }

  return config;
}

// Helper function to create class name to ID map
std::unordered_map<std::string, int> CreateClassMap() {
  return {{"person", 0}, {"car", 2}, {"dog", 16}, {"cat", 15}, {"bicycle", 1}};
}

class PolygonProcessorTest : public ::testing::Test {
 protected:
  std::unordered_map<std::string, int> class_map_;

  void SetUp() override { class_map_ = CreateClassMap(); }
};

// Test 1: Basic INCLUDE polygon - point inside should be included
TEST_F(PolygonProcessorTest, BasicIncludePolygon) {
  std::vector<exchange_protocol::PolygonConfig> polygons;
  polygons.push_back(CreatePolygonConfig(
      {{100, 100}, {400, 100}, {400, 400}, {100, 400}},
      exchange_protocol::PolygonType::INCLUDE, 1, {"person", "car"}));

  PolygonProcessor processor(polygons, class_map_);

  // Point inside polygon, class "person" (id=0)
  EXPECT_TRUE(processor.IsPointInPolygons(cv::Point(250, 250), 0));

  // Point inside polygon, class "car" (id=2)
  EXPECT_TRUE(processor.IsPointInPolygons(cv::Point(250, 250), 2));

  // Point outside polygon, class "person"
  EXPECT_FALSE(processor.IsPointInPolygons(cv::Point(50, 50), 0));
}

// Test 2: Basic EXCLUDE polygon - point inside should be excluded
TEST_F(PolygonProcessorTest, BasicExcludePolygon) {
  std::vector<exchange_protocol::PolygonConfig> polygons;
  polygons.push_back(CreatePolygonConfig(
      {{100, 100}, {400, 100}, {400, 400}, {100, 400}},
      exchange_protocol::PolygonType::EXCLUDE, 1, {"person"}));

  PolygonProcessor processor(polygons, class_map_);

  // Point inside EXCLUDE polygon should be excluded
  EXPECT_FALSE(processor.IsPointInPolygons(cv::Point(250, 250), 0));

  // Point outside polygon should also be excluded (not in any INCLUDE)
  EXPECT_FALSE(processor.IsPointInPolygons(cv::Point(50, 50), 0));
}

// Test 3: Class filter - only specified classes should be affected
TEST_F(PolygonProcessorTest, ClassFilter) {
  std::vector<exchange_protocol::PolygonConfig> polygons;
  polygons.push_back(CreatePolygonConfig(
      {{100, 100}, {400, 100}, {400, 400}, {100, 400}},
      exchange_protocol::PolygonType::INCLUDE, 1,
      {"person"}  // Only person class
      ));

  PolygonProcessor processor(polygons, class_map_);

  cv::Point inside_point(250, 250);

  // Class "person" (id=0) should be included
  EXPECT_TRUE(processor.IsPointInPolygons(inside_point, 0));

  // Class "car" (id=2) should NOT be included (not in class_filters)
  EXPECT_FALSE(processor.IsPointInPolygons(inside_point, 2));

  // Class "dog" (id=16) should NOT be included
  EXPECT_FALSE(processor.IsPointInPolygons(inside_point, 16));
}

// Test 4: Empty class_filters - polygon doesn't apply to any class
TEST_F(PolygonProcessorTest, EmptyClassFilters) {
  std::vector<exchange_protocol::PolygonConfig> polygons;
  polygons.push_back(CreatePolygonConfig(
      {{100, 100}, {400, 100}, {400, 400}, {100, 400}},
      exchange_protocol::PolygonType::INCLUDE, 1, {}  // Empty class_filters
      ));

  PolygonProcessor processor(polygons, class_map_);

  cv::Point inside_point(250, 250);

  // No class should be affected by polygon with empty class_filters
  EXPECT_FALSE(processor.IsPointInPolygons(inside_point, 0));
  EXPECT_FALSE(processor.IsPointInPolygons(inside_point, 2));
  EXPECT_FALSE(processor.IsPointInPolygons(inside_point, 16));
}

// Test 5: Priority - higher priority polygon overrides lower priority
TEST_F(PolygonProcessorTest, PriorityHandling) {
  std::vector<exchange_protocol::PolygonConfig> polygons;

  // Large INCLUDE polygon with priority 1
  polygons.push_back(CreatePolygonConfig(
      {{50, 50}, {450, 50}, {450, 450}, {50, 450}},
      exchange_protocol::PolygonType::INCLUDE, 1, {"person"}));

  // Small EXCLUDE polygon inside, with priority 2 (higher)
  polygons.push_back(CreatePolygonConfig(
      {{200, 200}, {300, 200}, {300, 300}, {200, 300}},
      exchange_protocol::PolygonType::EXCLUDE, 2, {"person"}));

  PolygonProcessor processor(polygons, class_map_);

  // Point in INCLUDE zone only (outside small EXCLUDE)
  EXPECT_TRUE(processor.IsPointInPolygons(cv::Point(100, 100), 0));

  // Point in overlap zone - EXCLUDE with higher priority wins
  EXPECT_FALSE(processor.IsPointInPolygons(cv::Point(250, 250), 0));
}

// Test 6: Same priority - EXCLUDE takes precedence over INCLUDE
TEST_F(PolygonProcessorTest, SamePriorityExcludePrecedence) {
  std::vector<exchange_protocol::PolygonConfig> polygons;

  // INCLUDE polygon with priority 1
  polygons.push_back(CreatePolygonConfig(
      {{100, 100}, {400, 100}, {400, 400}, {100, 400}},
      exchange_protocol::PolygonType::INCLUDE, 1, {"person"}));

  // Overlapping EXCLUDE polygon with same priority 1
  polygons.push_back(CreatePolygonConfig(
      {{150, 150}, {450, 150}, {450, 450}, {150, 450}},
      exchange_protocol::PolygonType::EXCLUDE, 1, {"person"}));

  PolygonProcessor processor(polygons, class_map_);

  // Point in overlap - EXCLUDE should win
  EXPECT_FALSE(processor.IsPointInPolygons(cv::Point(250, 250), 0));

  // Point only in INCLUDE (not in EXCLUDE)
  EXPECT_TRUE(processor.IsPointInPolygons(cv::Point(120, 120), 0));
}

// Test 7: Multiple overlapping polygons with different priorities
TEST_F(PolygonProcessorTest, ComplexOverlapping) {
  std::vector<exchange_protocol::PolygonConfig> polygons;

  // Layer 1: Large INCLUDE, priority 1
  polygons.push_back(CreatePolygonConfig(
      {{0, 0}, {500, 0}, {500, 500}, {0, 500}},
      exchange_protocol::PolygonType::INCLUDE, 1, {"person", "car"}));

  // Layer 2: Medium EXCLUDE, priority 2
  polygons.push_back(CreatePolygonConfig(
      {{100, 100}, {400, 100}, {400, 400}, {100, 400}},
      exchange_protocol::PolygonType::EXCLUDE, 2, {"person"}));

  // Layer 3: Small INCLUDE, priority 3 (highest)
  polygons.push_back(CreatePolygonConfig(
      {{200, 200}, {300, 200}, {300, 300}, {200, 300}},
      exchange_protocol::PolygonType::INCLUDE, 3, {"person"}));

  PolygonProcessor processor(polygons, class_map_);

  // Zone 1: Only in layer 1 (INCLUDE)
  EXPECT_TRUE(processor.IsPointInPolygons(cv::Point(50, 50), 0));

  // Zone 2: In layer 1 and 2 (EXCLUDE wins)
  EXPECT_FALSE(processor.IsPointInPolygons(cv::Point(150, 150), 0));

  // Zone 3: In all layers (layer 3 INCLUDE wins - highest priority)
  EXPECT_TRUE(processor.IsPointInPolygons(cv::Point(250, 250), 0));

  // Test "car" class - not affected by layer 2 and 3
  EXPECT_TRUE(processor.IsPointInPolygons(cv::Point(50, 50), 2));
  EXPECT_TRUE(processor.IsPointInPolygons(cv::Point(150, 150), 2));
  EXPECT_TRUE(processor.IsPointInPolygons(cv::Point(250, 250), 2));
}

// Test 8: Point on polygon boundary
TEST_F(PolygonProcessorTest, PointOnBoundary) {
  std::vector<exchange_protocol::PolygonConfig> polygons;
  polygons.push_back(CreatePolygonConfig(
      {{100, 100}, {400, 100}, {400, 400}, {100, 400}},
      exchange_protocol::PolygonType::INCLUDE, 1, {"person"}));

  PolygonProcessor processor(polygons, class_map_);

  // Point exactly on boundary should be considered inside
  EXPECT_TRUE(processor.IsPointInPolygons(cv::Point(100, 100), 0));
  EXPECT_TRUE(processor.IsPointInPolygons(cv::Point(250, 100), 0));
}

// Test 9: Invalid polygon (less than 3 points) should throw
TEST_F(PolygonProcessorTest, InvalidPolygonThrows) {
  std::vector<exchange_protocol::PolygonConfig> polygons;
  polygons.push_back(CreatePolygonConfig(
      {{100, 100}, {200, 200}},  // Only 2 points
      exchange_protocol::PolygonType::INCLUDE, 1, {"person"}));

  EXPECT_THROW(
      { PolygonProcessor processor(polygons, class_map_); },
      std::invalid_argument);
}

// Test 10: Triangular polygon
TEST_F(PolygonProcessorTest, TriangularPolygon) {
  std::vector<exchange_protocol::PolygonConfig> polygons;
  polygons.push_back(CreatePolygonConfig(
      {{250, 100}, {400, 400}, {100, 400}},  // Triangle
      exchange_protocol::PolygonType::INCLUDE, 1, {"person"}));

  PolygonProcessor processor(polygons, class_map_);

  // Point inside triangle
  EXPECT_TRUE(processor.IsPointInPolygons(cv::Point(250, 300), 0));

  // Point outside triangle
  EXPECT_FALSE(processor.IsPointInPolygons(cv::Point(100, 100), 0));
}

// Test 11: Non-convex (concave) polygon
TEST_F(PolygonProcessorTest, NonConvexPolygon) {
  std::vector<exchange_protocol::PolygonConfig> polygons;

  // L-shaped polygon
  polygons.push_back(CreatePolygonConfig(
      {{100, 100}, {300, 100}, {300, 200}, {200, 200}, {200, 300}, {100, 300}},
      exchange_protocol::PolygonType::INCLUDE, 1, {"person"}));

  PolygonProcessor processor(polygons, class_map_);

  // Point in the L-shape
  EXPECT_TRUE(processor.IsPointInPolygons(cv::Point(150, 150), 0));
  EXPECT_TRUE(processor.IsPointInPolygons(cv::Point(150, 250), 0));

  // Point in the "hole" of L-shape
  EXPECT_FALSE(processor.IsPointInPolygons(cv::Point(250, 250), 0));
}

// Test 12: Unknown class name in filter (should be ignored)
TEST_F(PolygonProcessorTest, UnknownClassInFilter) {
  std::vector<exchange_protocol::PolygonConfig> polygons;
  polygons.push_back(
      CreatePolygonConfig({{100, 100}, {400, 100}, {400, 400}, {100, 400}},
                          exchange_protocol::PolygonType::INCLUDE, 1,
                          {"person", "unknown_class", "car"}));

  // Should not throw, unknown class is simply ignored
  EXPECT_NO_THROW({
    PolygonProcessor processor(polygons, class_map_);

    // Known classes should still work
    EXPECT_TRUE(processor.IsPointInPolygons(cv::Point(250, 250), 0));
    EXPECT_TRUE(processor.IsPointInPolygons(cv::Point(250, 250), 2));
  });
}

// Test 13: Multiple non-overlapping polygons
TEST_F(PolygonProcessorTest, MultipleNonOverlappingPolygons) {
  std::vector<exchange_protocol::PolygonConfig> polygons;

  // Polygon 1: INCLUDE for person
  polygons.push_back(CreatePolygonConfig(
      {{100, 100}, {200, 100}, {200, 200}, {100, 200}},
      exchange_protocol::PolygonType::INCLUDE, 1, {"person"}));

  // Polygon 2: INCLUDE for car
  polygons.push_back(
      CreatePolygonConfig({{300, 300}, {400, 300}, {400, 400}, {300, 400}},
                          exchange_protocol::PolygonType::INCLUDE, 1, {"car"}));

  PolygonProcessor processor(polygons, class_map_);

  // Person in polygon 1
  EXPECT_TRUE(processor.IsPointInPolygons(cv::Point(150, 150), 0));
  // Car in polygon 1 (should be false - wrong class)
  EXPECT_FALSE(processor.IsPointInPolygons(cv::Point(150, 150), 2));

  // Car in polygon 2
  EXPECT_TRUE(processor.IsPointInPolygons(cv::Point(350, 350), 2));
  // Person in polygon 2 (should be false - wrong class)
  EXPECT_FALSE(processor.IsPointInPolygons(cv::Point(350, 350), 0));

  // Point outside both polygons
  EXPECT_FALSE(processor.IsPointInPolygons(cv::Point(250, 250), 0));
  EXPECT_FALSE(processor.IsPointInPolygons(cv::Point(250, 250), 2));
}

}  // namespace

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}