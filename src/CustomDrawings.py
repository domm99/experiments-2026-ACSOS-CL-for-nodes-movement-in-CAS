from phyelds.simulator.effects import Effect
from abc import ABC, abstractmethod
from enum import Enum
from typing import Literal, Optional, Any, Annotated, Tuple, List
from pydantic import BaseModel, Field, BeforeValidator, SerializeAsAny
from phyelds.simulator import Environment
from phyelds.simulator.effects import Link

class CustomDrawEdges(Effect):
    """
    Draw edges between nodes.
    """
    type: Literal["DrawEdges"] = "DrawEdges"
    alpha: float = 0.9
    z_order: int = 0

    def apply(self, ax, environment: Environment):
        """
        Draw edges between nodes.
        """
        all_neighbors_tuple = set()
        for node in environment.nodes.values():
            neighbors = node.get_neighbors()
            for neighbor in neighbors:
                all_neighbors_tuple.add(Link(node.position, neighbor.position))
        for link in all_neighbors_tuple:
            ax.plot(
                [link.node1[0], link.node2[0]],
                [link.node1[1], link.node2[1]],
                alpha=self.alpha,
                color="gray",
                zorder=self.z_order,
                linewidth=4
            )


class CustomDrawNodes(Effect):
    """
    Draw nodes.
    """
    type: Literal["DrawNodes"] = "DrawNodes"
    color_from: Optional[str] = None
    z_order: int = 10

    def apply(self, ax, environment: Environment):
        """
        Draw nodes.
        """
        positions = [node.position for node in environment.nodes.values()]
        if not positions:
            return
        x, y = zip(*positions)

        if self.color_from:
            colors = [
                node.data.get(self.color_from, "blue")
                for node in environment.nodes.values()
            ]
            ax.scatter(x, y, c=colors, zorder=self.z_order, s=150)
        else:
            ax.scatter(x, y, c="blue", zorder=self.z_order, s=150)

























