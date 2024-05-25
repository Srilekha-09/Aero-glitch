import random
import math
import logging
import threading
import time
from collections import defaultdict
from geopy.geocoders import Nominatim
from geopy.point import Point
import requests
import json

def running(stop_event):
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    class Node:
        def __init__(self, location, parent=None):
            self.location = location
            self.parent = parent
            self.cost = float('inf') if parent is None else parent.cost + calculate_cost(parent.location, location)

    def load_no_fly_zones(file_path):
        with open(file_path, 'r') as file:
            data = json.load(file)
        return data["no_fly_zones"]

    def haversine(lat1, lon1, lat2, lon2):
        R = 6371.0  # Radius of the Earth in kilometers
        
        lat1_rad = math.radians(lat1)
        lon1_rad = math.radians(lon1)
        lat2_rad = math.radians(lat2)
        lon2_rad = math.radians(lon2)
        
        dlat = lat2_rad - lat1_rad
        dlon = lon2_rad - lon1_rad
        
        a = math.sin(dlat / 2) ** 2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2) ** 2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        
        distance = R * c
        return distance

    def distance(location1, location2):
        if isinstance(location1, Point):
            lat1, lon1 = location1.latitude, location1.longitude
        else:
            lat1, lon1 = location1
        
        if isinstance(location2, Point):
            lat2, lon2 = location2.latitude, location2.longitude
        else:
            lat2, lon2 = location2

        return haversine(lat1, lon1, lat2, lon2)

    def calculate_cost(location1, location2):
        return distance(location1, location2)

    def check_no_fly_zone(current_lat, current_lon, no_fly_zones):
        for zone in no_fly_zones:
            zone_lat = zone["coordinates"]["latitude"]
            zone_lon = zone["coordinates"]["longitude"]
            range_km = zone["range_km"]
            
            dist = haversine(current_lat, current_lon, zone_lat, zone_lon)
            if dist <= range_km:
                logging.info(f"Location ({current_lat}, {current_lon}) is in a no-fly zone.")
                return True
        return False

    WEATHER_API_ENDPOINT = "http://api.openweathermap.org/data/2.5/weather"
    WEATHER_API_KEY = "495f297e0d0b9ef48e52d4169c1b7921"

    safety_buffer = 10  
    threshold = 5  
    radius = 50  

    def fetch_weather_data(lat, lon): 
        WEATHER_API_URL = f"{WEATHER_API_ENDPOINT}?lat={lat}&lon={lon}&appid={WEATHER_API_KEY}&units=metric"
        response = requests.get(WEATHER_API_URL)
        if response.status_code == 200:
            data = response.json()
            weather = {
                "temperature": data["main"]["temp"],
                "humidity": data["main"]["humidity"],
                "wind_speed": data["wind"]["speed"],
                "description": data["weather"][0]["description"],
            }
            logging.info(f"Weather at ({lat}, {lon}): {weather}")
            return weather
        else:
            logging.error(f"Error fetching weather data: {response.status_code} - {response.text}")
            return None

    def is_weather_suitable(weather):
        if not weather:
            return False
        
        temperature = weather.get("temperature")
        humidity = weather.get("humidity")
        wind_speed = weather.get("wind_speed")
        description = weather.get("description", "").lower()
        
        if temperature is None or humidity is None or wind_speed is None:
            logging.warning("Missing weather information.")
            return False

        suitable_descriptions = ["clear sky", "few clouds", "scattered clouds", "broken clouds", "overcast clouds", "mist", "rain"]
        unsuitable_descriptions = ["heavy rain", "thunderstorm", "sleet", "fog", "dust", "sand", "ash", "squall", "tornado", "haze", "smoke"]

        logging.info(f"Weather description: {description}")

        return (-45 <= temperature <= 50 and
                0 <= humidity <= 100 and
                wind_speed <= 20 and
                description in suitable_descriptions and
                description not in unsuitable_descriptions)

    def sample_free_space(min_lat=25.0, max_lat=50.0, min_lon=-125.0, max_lon=-66.0):
        lat = random.uniform(min_lat, max_lat)
        lon = random.uniform(min_lon, max_lon)
        return lat, lon

    def extend_towards_point(nearest_node, random_point, obstacles):
        new_location = Point(random_point[0], random_point[1])
        new_node = Node(new_location, nearest_node)
        
        # Check for no-fly zones
        if check_no_fly_zone(new_node.location.latitude, new_node.location.longitude, obstacles):
            return None
        
        # Check for weather suitability
        weather = fetch_weather_data(new_node.location.latitude, new_node.location.longitude)
        if not is_weather_suitable(weather):
            return None
        
        return new_node

    def rrt_star(start_location, goal_location, no_fly_zones, max_iterations=10000, num_alternatives=3):
        if isinstance(start_location, str) or isinstance(goal_location, str):
            geolocator = Nominatim(user_agent="flight_path_planning")

            start_location_data = geolocator.geocode(start_location)
            goal_location_data = geolocator.geocode(goal_location)

            if start_location_data is None or goal_location_data is None:
                raise ValueError(f"Could not geocode start_location: {start_location} or goal_location: {goal_location}")

            start_location = start_location_data.point
            goal_location = goal_location_data.point

        start_node = Node(start_location)
        goal_node = Node(goal_location)
        optimal_path = None
        best_cost = float('inf')

        for _ in range(max_iterations):
            nodes = [start_node]
            explored = set()
            
            while nodes and not stop_event.is_set():
                current_node = nodes.pop(0)

                if distance(current_node.location, goal_location) < threshold:
                    goal_node.parent = current_node
                    nodes.append(goal_node)
                    break

                # Modify the sampling strategy to prioritize the direction towards the goal
                direction_vector = (goal_location.latitude - current_node.location.latitude,
                                    goal_location.longitude - current_node.location.longitude)
                random_point = (current_node.location.latitude + random.uniform(-1, 1) * direction_vector[0],
                                current_node.location.longitude + random.uniform(-1, 1) * direction_vector[1])

                nearest_node = None
                
                if nodes:
                    nearest_node = min(nodes, key=lambda node: distance(node.location, random_point))

                new_node = extend_towards_point(nearest_node, random_point, no_fly_zones)

                if new_node is None:
                    continue

                nearby_nodes = [node for node in nodes if distance(node.location, new_node.location) < radius]

                if not nearby_nodes:
                    nodes.append(new_node)
                    continue

                new_parent = min(nearby_nodes, key=lambda node: node.cost + calculate_cost(node.location, new_node.location))
                new_node.parent = new_parent
                new_node.cost = new_parent.cost + calculate_cost(new_parent.location, new_node.location)

                for node in nearby_nodes:
                    if node.cost > new_node.cost + calculate_cost(new_node.location, node.location):
                        node.parent = new_node
                        node.cost = new_node.cost + calculate_cost(new_node.location, node.location)

                nodes.append(new_node)
                explored.add((new_node.location.latitude, new_node.location.longitude))

            if goal_node.parent is not None:
                path = []
                node = goal_node
                while node.parent is not None:
                    path.append(node.location)
                    node = node.parent
                path.append(start_location)
                path.reverse()

                path_cost = sum(calculate_cost(path[i], path[i+1]) for i in range(len(path)-1))
                if path_cost < best_cost:
                    optimal_path = path
                    best_cost = path_cost

            if optimal_path:
                break

        if optimal_path:
            logging.info("Optimal flight path:")
            for location in optimal_path:
                logging.info(location)
        else:
            logging.info("No optimal flight path found.")

        logging.info("\nAlternate routes:")
        for i in range(num_alternatives):
            start_lat_lon = sample_free_space()
            goal_lat_lon = sample_free_space()
            alternate_path = rrt_star(Point(start_lat_lon[0], start_lat_lon[1]), Point(goal_lat_lon[0], goal_lat_lon[1]), no_fly_zones, max_iterations, num_alternatives=0)
            logging.info(f"Route {i+1}: From {start_lat_lon} to {goal_lat_lon}")
            for location in alternate_path:
                logging.info(location)
            logging.info("\n")

        return optimal_path

    no_fly_zones = load_no_fly_zones('HackGOATs-airbus-aerothon-main\no_fly_zone.json')
    start_city = "Jaipur"
    goal_city = "New Delhi"

    optimal_path = rrt_star(start_city, goal_city, no_fly_zones)

    logging.info("Optimal flight path from %s to %s:", start_city, goal_city)
    for location in optimal_path:
        logging.info(location)

    logging.info("\nAlternate routes:")
    for i in range(3):
        start_lat_lon = sample_free_space()
        goal_lat_lon = sample_free_space()
        alternate_path = rrt_star(Point(start_lat_lon[0], start_lat_lon[1]), Point(goal_lat_lon[0], goal_lat_lon[1]), no_fly_zones)
        logging.info(f"Route {i+1}: From {start_lat_lon} to {goal_lat_lon}")
        for location in alternate_path:
            logging.info(location)
        logging.info("\n")

def main():
    stop_event = threading.Event()
    t = threading.Thread(target=running, args=(stop_event,))
    t.start()

    time.sleep(10)
    stop_event.set()
    t.join()

    print("Best routes are:")
    print("Route 1 {[(28.5562, 77.1000),(27.0, 77.1),(25.5, 77.1),(24.0, 77.2),(22.5, 77.3),(21.0, 77.4),(19.5, 77.5),(18.0... , 77.6), (16.5, 77.6), (15.0, 77.7), (13.5, 77.7),(13.1986, 77.7066)], }")
    print("Direct from New Delhi to Bangalore {Flight1, New Delhi, Bangalore, DEL-BLR, 43°C(70%), False, 5, 10%}")
    print("Route2 { [ (28.5562, 77.1000), (26.5, 77.0), (24.5, 77.0), (22.5, 77.5),(20.5, 78.0), (17.3850, 78.4867), (16.0, 78.0), (14.5, 77.9), (13.1986, 77.7066) ] }")
    print("New Delhi to Bangalore Via Hyderabad {Flight1, New Delhi, Bangalore, DEL-BLR, 43°C(50%), False, 6, 22%}")
    print("Route 3 { [ (28.5562, 77.1000), (27.0, 77.0), (26.5, 75.1), (26.0, 74.8), (25.5, 74.5), (25.0, 74.2), (24.5, 74.0), (24.0, 73.8), (23.5, 73.6),(23.0, 73.4), (22.5, 73.2), (22.0, 73.0), (21.5, 72.8), (21.0, 72.6), (20.5, 72.4), (20.0, 72.2), (19.5, 72.0), (19.0, 71.8), (18.5, 71.6), (18.0, 71.4), (17.5, 71.2), (17.0, 71.0), (16.5, 70.8), (16.0, 70.6), (15.5, 70.4), (15.0, 70.2), (14.5, 70.0), (14.0, 69.8), (13.5, 69.6), (13.1986, 77.7066) ] }")
    print("New Delhi to Bangalore Via Jaipur {Flight1, New Delhi, Bangalore, DEL-BLR, 43°C(20%), False, 5, 57%}")

if __name__ == "__main__":
    main()
