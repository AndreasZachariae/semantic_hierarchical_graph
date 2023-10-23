##############################################################################
##                                 Base Image                               ##
##############################################################################
ARG ROS_DISTRO=foxy
FROM ros:$ROS_DISTRO-ros-base
ENV TZ=Europe/Berlin
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone
RUN rosdep update --rosdistro $ROS_DISTRO

# Update packages only if necessary, ~250MB
# RUN apt update && apt -y dist-upgrade

##############################################################################
##                                 Global Dependecies                       ##
##############################################################################
RUN apt-get update && apt-get install --no-install-recommends -y \
    python3-pip \
    ffmpeg libsm6 libxext6  \
    python3-colcon-common-extensions \
    ros-$ROS_DISTRO-navigation2 \
    ros-$ROS_DISTRO-nav2-* \
    ros-$ROS_DISTRO-turtlebot3-* \
    ros-$ROS_DISTRO-rviz2 \
    ros-$ROS_DISTRO-gazebo-* \
    ros-$ROS_DISTRO-tf-transformations \
    ros-$ROS_DISTRO-rqt* \
    ros-$ROS_DISTRO-slam-toolbox \
    ros-$ROS_DISTRO-teleop-twist-joy \
    ros-$ROS_DISTRO-teleop-twist-keyboard \
    xterm \
    qtbase5-dev \
    libqt5svg5-dev \
    libzmq3-dev \
    libdw-dev \
    libqt5opengl5-dev \
    qttools5-dev-tools \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install -U \
    pip \
    setuptools \
    pipenv \
    transforms3d

##############################################################################
##                                 Create User                              ##
##############################################################################
ARG USER=docker
ARG PASSWORD=docker
ARG UID=1000
ARG GID=1000
ENV UID=$UID
ENV GID=$GID
ENV USER=$USER
RUN groupadd -g "$GID" "$USER"  && \
    useradd -m -u "$UID" -g "$GID" --shell $(which bash) "$USER" -G sudo && \
    echo "$USER:$PASSWORD" | chpasswd && \
    echo "%sudo ALL=(ALL) NOPASSWD: ALL" > /etc/sudoers.d/sudogrp
RUN echo "source /opt/ros/$ROS_DISTRO/setup.bash" >> /etc/bash.bashrc

USER $USER 
RUN mkdir -p /home/$USER/ros2_ws/src

##############################################################################
##                                 User Dependecies                         ##
##############################################################################
# For aws-hospital-simulation
WORKDIR /home/$USER/ros2_ws/src/
# RUN git clone -b ros2 https://github.com/AndreasZachariae/aws-robomaker-hospital-world.git
COPY ./aws-robomaker-hospital-world ./aws-robomaker-hospital-world
# RUN python3 -m pip install -r aws-robomaker-hospital-world/requirements.txt
# RUN python3 aws-robomaker-hospital-world/fuel_utility.py download -m XRayMachine -m IVStand -m BloodPressureMonitor -m BPCart -m BMWCart -m CGMClassic -m StorageRack -m Chair -m InstrumentCart1 -m Scrubs -m PatientWheelChair -m WhiteChipChair -m TrolleyBed -m SurgicalTrolley -m PotatoChipChair -m VisitorKidSit -m FemaleVisitorSit -m AdjTable -m MopCart3 -m MaleVisitorSit -m Drawer -m OfficeChairBlack -m ElderLadyPatient -m ElderMalePatient -m InstrumentCart2 -m MetalCabinet -m BedTable -m BedsideTable -m AnesthesiaMachine -m TrolleyBedPatient -m Shower -m SurgicalTrolleyMed -m StorageRackCovered -m KitchenSink -m Toilet -m VendingMachine -m ParkingTrolleyMin -m PatientFSit -m MaleVisitorOnPhone -m FemaleVisitor -m MalePatientBed -m StorageRackCoverOpen -m ParkingTrolleyMax \
#     -d aws-robomaker-hospital-world/fuel_models --verbose

COPY ./petra_description ./petra_description

RUN git clone --branch $ROS_DISTRO      https://github.com/neobotix/neo_simulation2.git
RUN git clone --branch galactic         https://github.com/neobotix/neo_nav2_bringup.git
RUN git clone --branch $ROS_DISTRO      https://github.com/neobotix/neo_local_planner2.git
RUN git clone --branch galactic         https://github.com/neobotix/neo_localization2.git
RUN git clone --branch master           https://github.com/neobotix/neo_common2
RUN git clone --branch master           https://github.com/neobotix/neo_msgs2
RUN git clone --branch master           https://github.com/neobotix/neo_srvs2

# For semantic_hierarchical_graph
WORKDIR /home/$USER/ros2_ws/src/semantic_hierarchical_graph

COPY ./Pipfile ./Pipfile
COPY ./Pipfile.lock ./Pipfile.lock
RUN pipenv install --system --deploy --ignore-pipfile

COPY ./config ./config
COPY ./data/graphs ./data/graphs
COPY ./path_planner_suite ./path_planner_suite
COPY ./ros2 ./ros2
COPY ./ros2_interfaces ./ros2_interfaces
COPY ./semantic_hierarchical_graph ./semantic_hierarchical_graph

ENV TURTLEBOT3_MODEL=waffle
ENV GAZEBO_MODEL_PATH=$GAZEBO_MODEL_PATH:/opt/ros/$ROS_DISTRO/share/turtlebot3_gazebo/models:/home/$USER/ros2_ws/src/aws-robomaker-hospital-world/fuel_models:/home/$USER/ros2_ws/src/aws-robomaker-hospital-world/models

##############################################################################
##                                 Build ROS and run                        ##
##############################################################################
WORKDIR /home/$USER/ros2_ws
RUN . /opt/ros/$ROS_DISTRO/setup.sh && colcon build --symlink-install
RUN echo "source /home/$USER/ros2_ws/install/setup.bash" >> /home/$USER/.bashrc

RUN sudo sed --in-place --expression \
    '$isource "/home/$USER/ros2_ws/install/setup.bash"' \
    /ros_entrypoint.sh

RUN sudo sed --in-place --expression \
    '$isource "/usr/share/gazebo/setup.sh"' \
    /ros_entrypoint.sh

CMD /bin/bash
# CMD ["ros2", "launch", "shg", "graph.launch.py"]
# CMD ["ros2", "run", "shg", "graph_node"]