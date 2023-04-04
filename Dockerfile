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
    ros-$ROS_DISTRO-navigation2 \
    ros-$ROS_DISTRO-nav2-bringup \
    ros-$ROS_DISTRO-turtlebot3-* \
    ros-$ROS_DISTRO-rviz2 \
    ros-$ROS_DISTRO-gazebo-* \
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
    pipenv

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
RUN git clone -b ros2 https://github.com/AndreasZachariae/aws-robomaker-hospital-world.git
RUN python3 -m pip install -r aws-robomaker-hospital-world/requirements.txt
RUN python3 aws-robomaker-hospital-world/fuel_utility.py download -m XRayMachine -m IVStand -m BloodPressureMonitor -m BPCart -m BMWCart -m CGMClassic -m StorageRack -m Chair -m InstrumentCart1 -m Scrubs -m PatientWheelChair -m WhiteChipChair -m TrolleyBed -m SurgicalTrolley -m PotatoChipChair -m VisitorKidSit -m FemaleVisitorSit -m AdjTable -m MopCart3 -m MaleVisitorSit -m Drawer -m OfficeChairBlack -m ElderLadyPatient -m ElderMalePatient -m InstrumentCart2 -m MetalCabinet -m BedTable -m BedsideTable -m AnesthesiaMachine -m TrolleyBedPatient -m Shower -m SurgicalTrolleyMed -m StorageRackCovered -m KitchenSink -m Toilet -m VendingMachine -m ParkingTrolleyMin -m PatientFSit -m MaleVisitorOnPhone -m FemaleVisitor -m MalePatientBed -m StorageRackCoverOpen -m ParkingTrolleyMax \
    -d aws-robomaker-hospital-world/fuel_models --verbose

# For semantic_hierarchical_graph
WORKDIR /home/$USER/ros2_ws/src/semantic_hierarchical_graph

COPY ./Pipfile ./Pipfile
COPY ./Pipfile.lock ./Pipfile.lock
RUN pipenv install --system --deploy --ignore-pipfile

COPY ./config ./config
COPY ./data/graphs ./data/graphs
COPY ./path_planner_suite ./path_planner_suite
COPY ./ros2 ./ros2
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

CMD /bin/bash
# CMD ["ros2", "launch", "shg", "graph.launch.py"]
# CMD ["ros2", "run", "shg", "graph_node"]