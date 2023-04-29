import random
import os
from time import sleep
from copy import deepcopy
from functools import reduce


class Node:
    """A class representing a single node participating in the simulation.
    Attributes
    ----------
    name: str
        the displayed name of the node (aliased as the first character)
    position: int
        the placement of the node's connection to the cable (should be >= 0 and  < cable size)
    collision: bool
        indicates that a collision has taken place while the node was broadcasting
    collision_count: int
        the number of times a collision occurred while trying to send the current frame
    broadcasting: bool
        indicates that the node is currently broadcasting something (whether it's jam or real data)
    wait: int
        the number of iterations the node will stay in its current state for
    left: int
        the number of frames still to be broadcasted
    Stats
    -----
    total_collisions: int
        counts collisions through the entire simulation
    total_waiting_time: int
        counts time spent on waiting (after a collision or while the cable was busy)
    """

    def __init__(self, name, pos, start, to_send):
        self.name = str(name)
        self.position = pos
        self.collision = False
        self.collision_count = 0
        self.broadcasting = False
        self.wait = start
        self.left = to_send

        # Stats
        self.total_collisions = 0
        self.total_waiting_time = 0
        self.total_successful_transmissions = 0

    def flag(self) -> str:
        """Return the node's state in a string form (Idle, Waiting, Collided or Active)."""
        if self.broadcasting:
            return "Collided" if self.collision else "Active"
        else:
            return "Idle" if self.left == 0 else "Waiting"

    def reschedule(self):
        """Increment the collision count and choose the amount of time to wait before trying to rebroadcast."""
        self.collision_count += 1
        self.wait = random.randrange(0, 2 ** min((self.collision_count, 10)))
        self.total_waiting_time += self.wait


class Signal:
    """Represents a signal contained in a single fragment of the shared cable
    Attributes
    ----------
    node: Node
        the source node
    direction: int
        determines in which direction will the signal propagate
        -1: left
         1: right
         0: both
    jam: bool
        determines whether the signal is a jam signal
    """

    def __init__(self, node, direction):
        self.node = node
        self.direction = direction
        self.jam = node.collision

    def __repr__(self):
        if self.jam:
            return self.node.name[0] + "*"
        return self.node.name[0]

    def __str__(self):
        return repr(self)


class Simulation:
    """A class simulating the CSMA/CD protocol.
    Attributes
    ----------
    nodes: list
        a list of all nodes
    active_nodes: list
        a list of non-idle nodes
    cable: list
        represents the cable connecting the nodes
    empty_cable: list
        used to reset the cable while evaluating how signals propagate
    size:
        the number of cable segments, ie. the maximum number of nodes that can be added
    UI-specific attributes
    ----------------------
    names: list
        a list of node names with respect to the position
    width: int
        column width
    log: list
        a list of messages generated while simulating
    """

    def __init__(self, size):
        self.nodes = {i: None for i in range(size)}
        self.active_nodes = []
        self.cable: list[list[Signal]] = [[] for _ in range(size)]
        self.empty_cable = deepcopy(self.cable)
        self.size = size

        self.names = ["" for _ in range(self.size)]
        self.width = 12
        self.log = []
        self.max_propagation_time = 2 * self.size - 2

    def add_node(self, name, position, start=0, frames_to_send=0):
        node = Node(name, position, start, frames_to_send)
        if position in self.nodes:
            self.nodes[position] = node
            self.names[position] = name
            if frames_to_send > 0:
                self.active_nodes.append(node)
        else:
            raise Exception(
                "Position has to be between 0 and {}.".format(self.size - 1)
            )

    def run(self, output_all=False, display_time=None):
        """Execute the simulation
        Arguments
        ---------
        output_all: bool
            if True, the entire simulation is printed at once; otherwise only state is shown at a time
        display_time: float
            if specified, the program will wait this much time before displaying the next state; 
            otherwise it waits for a keypress to continue
        """
        self.width = max((4 * (len(self.active_nodes)) - 2, 10))
        if output_all:
            self._print_header(True)
        i = -1
        while self.active_nodes or self.cable != self.empty_cable:
            i += 1
            self._step()
            if output_all:
                self._print_cable(i)
            else:
                self._print_state(i)
                if display_time:
                    sleep(display_time)
                else:
                    input("Press Enter to continue...")
        print(f"\n{i} iterations total. Node statistics:\n")
        for node in self.nodes.values():
            if node:
                print(node.name)
                print(f"successful transmissions: {node.total_successful_transmissions}")
                print(f"collisions: {node.total_collisions}")
                print(f"total waiting time: {node.total_waiting_time}\n")

    def _step(self):
        # Propagate signals present in the cable
        next_cable = deepcopy(self.empty_cable)
        for i, segment in enumerate(self.cable):
            for signal in segment:
                if signal.direction == -1:
                    if i > 0:
                        next_cable[i - 1].append(signal)
                elif signal.direction == 1:
                    if i < self.size - 1:
                        next_cable[i + 1].append(signal)
                else:
                    if i > 0:
                        next_cable[i - 1].append(Signal(signal.node, -1))
                    if i < self.size - 1:
                        next_cable[i + 1].append(Signal(signal.node, 1))

        self.cable = next_cable

        for node in self.active_nodes:
            # If possible, start broadcasting
            if not node.broadcasting:
                if node.wait == 0:
                    if not self.cable[node.position]:  # if not receiving anything
                        node.collision = False
                        self.log.append(f"{node.name} started broadcasting.")
                        self.cable[node.position].append(Signal(node, 0))
                        node.broadcasting = True
                        node.wait = self.max_propagation_time
                    else:  # if receiving sth
                        node.total_waiting_time += 1
                else:
                    if node.collision:
                        if not self.cable[node.position]:
                            node.wait -= 1
                            node.collision = False
                    elif not node.collision:
                        node.wait -= 1

            elif node.broadcasting:
                # if possible, end broadcasting
                if node.wait == 0:
                    node.broadcasting = False
                    if node.collision:
                        # node.collision = False
                        node.reschedule()  # sets new wait time
                        self.log.append(f"{node.name} is done jamming. "
                                        f"It will wait {node.wait} iteration(s) before trying to broadcast.")
                    else:  # it wasn't jamming
                        node.left -= 1
                        add_mes = (
                            "Its work is done."
                            if not node.left
                            else "{} more to go.".format(node.left)
                        )
                        self.log.append(f"{node.name} has successfully broadcast one of its frames. {add_mes}")
                        node.total_successful_transmissions += 1
                        node.total_collisions += node.collision_count
                        node.collision_count = 0
                        if node.left == 0:
                            self.active_nodes.remove(node)

                # Proceed to broadcast
                else:
                    # Collision detected
                    if not node.collision and len(self.cable[node.position]) > 0:
                        self.log.append(f"{node.name} has detected a collision. "
                                        f"It started to broadcast its jam signal.")
                        node.collision = True
                        node.wait = self.max_propagation_time
                    else:
                        node.wait -= 1

                    self.log.append(f"{node.name} continues to broadcast.")
                    self.cable[node.position].append(Signal(node, 0))

    # UI methods
    def _print_header(self, universal=False):
        times = ["" for _ in range(self.size)]
        flags = ["" for _ in range(self.size)]
        for node in self.nodes.values():
            if node:
                p = node.position
                times[p] = node.wait
                flags[p] = node.flag()

        w = self.width
        print(("+" + "-" * 6) + ("+" + "-" * w) * self.size + "+")
        print(
            "|{:^6}|".format("Name")
            + reduce(lambda a, b: "{:^{w}}|{:^{w}}".format(a, b, w=w), self.names)
            + "|"
        )
        if not universal:
            print(
                "|{:^6}|".format("Wait")
                + reduce(lambda a, b: "{:^{w}}|{:^{w}}".format(a, b, w=w), times)
                + "|"
            )
            print(
                "|{:^6}|".format("Flag")
                + reduce(lambda a, b: "{:^{w}}|{:^{w}}".format(a, b, w=w), flags)
                + "|"
            )
        print(("+" + "=" * 6) + ("+" + "=" * w) * (self.size) + "+")

    def _print_cable(self, iter):
        w = self.width
        signals = [", ".join(sorted(str(s) for s in frag)) for frag in self.cable]
        # print("|{:^10}".format("")*(self.size+1) + "|")
        print(
            "|{:^6}|".format("t={}".format(iter))
            + reduce(lambda a, b: "{:^{w}}|{:^{w}}".format(a, b, w=w), signals)
            + "|"
        )
        # print("|{:^10}".format(" ")*(self.size+1) + "|")
        print(("+" + "-" * 6) + ("+" + "-" * w) * self.size + "+")

    def _print_state(self, iter):
        # if os.name == "nt":
        #       os.system("cls")
        # else:
        #       os.system("clear")

        self._print_header()
        self._print_cable(iter)
        for mes in self.log:
            print(mes)
        self.log = []


if __name__ == "__main__":
    sim = Simulation(5)
    sim.add_node(name="A", position=0, start=0, frames_to_send=2)
    sim.add_node(name="B", position=2, start=0, frames_to_send=0)
    sim.add_node(name="C", position=4, start=3, frames_to_send=3)
    sim.run(output_all=True)
