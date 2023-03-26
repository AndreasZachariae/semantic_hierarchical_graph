import os
import time
import requests

flask_webserver_ip = os.environ["FLASK_BACKEND_URL"]


def wait_for_flask_webserver(max_wait_time=10):
    start_time = time.time()
    while start_time + max_wait_time > time.time():
        try:
            response = requests.get(flask_webserver_ip + "/")
            if response.status_code == 200:
                break
        except:
            print("Waiting for flask webserver to start...")
            time.sleep(1)


def get_emergency_state():
    response = requests.get(flask_webserver_ip + "/emergency_button")
    emergency_state = response.json()["state"]
    print(emergency_state)


def set_emergency_state():
    response = requests.post(flask_webserver_ip + "/emergency_button", json={"state": True})
    print(response.text)


def get_transport_start():
    response = requests.get(flask_webserver_ip + "/transport_start_button")
    transport_start = response.json()["state"]
    print(transport_start)


def send_new_floor_map():
    """ svg: floor map as svg
        origin: (x, y) pixel coordinates of the origin of the map
        resolution: (float) resolution of the map in meters per pixel
        floor: (int) floor number corresponding to hierarchical graph.
               Choose correct floor number based on robot position
    """
    with open("data/benchmark_maps/iras_slam2.svg", "r") as f:
        svg = f.read()
    locations = [{"position": [10, 10], "type": "elevator", "icon": "elevator.svg"},
                 {"position": [20, 20], "type": "info", "icon": "info.svg"},
                 {"position": [30, 30], "type": "door", "icon": "door.svg"}]

    response = requests.post(flask_webserver_ip + "/floor_map", json={"floor": 0,
                                                                      "svg": svg,
                                                                      "origin": [20, 20],
                                                                      "resolution": 0.03,
                                                                      "locations": locations})
    print(response.text)


def update_robot_pose():
    """ pose: (x, y, theta) pixel coordinates and angle in degrees of the robot
        floor: (int) floor number corresponding to hierarchical graph.
    """
    pose = [10, 10, 90]
    response = requests.post(flask_webserver_ip + "/robot_pose", json={"floor": 0,
                                                                       "pose": pose})
    print(response.text)


def send_new_robot_path():
    """ path: dict of hierarchical graph nodes to visit.
    """
    # Eventuell kann ich auch die Positionen als Liste pro floor ausgeben
    path_dict = {
        "floor_0": {
            "room_16": {
                "(88, 358)": {},
                "(191, 358)": {},
                "(193, 358)": {},
                "(200, 351)": {},
                "(203, 343)": {},
                "room_11_(203, 343)_bridge": {}
            },
            "room_11": {
                "room_16_(203, 343)_bridge": {},
                "(203, 343)": {},
                "(203, 278)": {},
                "(221, 278)": {},
            }
        }
    }
    response = requests.post(flask_webserver_ip + "/robot_path", json={"path": path_dict})
    print(response.text)


if __name__ == "__main__":
    wait_for_flask_webserver()
    get_emergency_state()
    set_emergency_state()
    get_transport_start()
    send_new_floor_map()
    update_robot_pose()
    send_new_robot_path()
