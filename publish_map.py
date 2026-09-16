#!/usr/bin/env python3
"""
publish_map.py - publica o mapa (map.yaml + map.pgm) como nav_msgs/OccupancyGrid.

Alternativa leve ao nav2_map_server: so usa rclpy, numpy e PyYAML.

Referencia: por defeito o mapa e publicado no frame "odom", que e o frame fixo
em que o /scan e desenhado (base_scan -> base_link -> base_footprint -> odom).
Nao se usa "base_scan" como frame do mapa porque esse frame anda com o robo.

Detalhes que fazem o mapa aparecer de forma fiavel no Foxglove:
  * use_sim_time = true  -> o stamp do mapa esta no tempo do bag (/clock),
    e nao em 2026 (wall clock), que ficaria "no futuro" face ao scan e as TF.
  * republica a cada `period` segundos (relogio de parede) -> se o Foxglove
    ligar mais tarde, ou se o bag for reiniciado (o tempo volta atras e o
    Foxglove limpa a cena), o mapa volta a aparecer.

Uso:
    python3 publish_map.py
    python3 publish_map.py --ros-args -p yaml:=/caminho/map.yaml
    python3 publish_map.py --ros-args -p frame_id:=map -p publish_tf:=true   # map -> odom identidade
"""

import os

import numpy as np
import yaml

import rclpy
from rclpy.clock import Clock, ClockType
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from geometry_msgs.msg import TransformStamped
from nav_msgs.msg import OccupancyGrid
from tf2_ros import StaticTransformBroadcaster


DEFAULT_YAML = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'map.yaml')


def read_pgm(path):
    """Le um PGM binario (P5) ou ASCII (P2) e devolve um array float em [0, 1] (linhas x colunas)."""
    with open(path, 'rb') as f:
        raw = f.read()

    # Header: magic, largura, altura, maxval (podem existir comentarios '#')
    tokens, i = [], 0
    while len(tokens) < 4:
        while raw[i:i + 1].isspace():
            i += 1
        if raw[i:i + 1] == b'#':
            while raw[i:i + 1] not in (b'\n', b''):
                i += 1
            continue
        j = i
        while not raw[j:j + 1].isspace():
            j += 1
        tokens.append(raw[i:j])
        i = j
    magic, width, height, maxval = tokens[0], int(tokens[1]), int(tokens[2]), int(tokens[3])

    if magic == b'P5':
        body = raw[i + 1:]  # exatamente um whitespace depois do maxval
        dtype = np.uint8 if maxval < 256 else '>u2'
        img = np.frombuffer(body, dtype=dtype, count=width * height)
    elif magic == b'P2':
        img = np.array(raw[i:].split()[:width * height], dtype=np.int32)
    else:
        raise ValueError(f'Formato PGM nao suportado: {magic!r}')

    return img.reshape(height, width).astype(np.float64) / maxval


def load_occupancy(yaml_path):
    """Converte o mapa para OccupancyGrid com as regras do map_server (modo trinary)."""
    with open(yaml_path) as f:
        meta = yaml.safe_load(f)

    image_path = meta['image']
    if not os.path.isabs(image_path):
        image_path = os.path.join(os.path.dirname(os.path.abspath(yaml_path)), image_path)

    img = read_pgm(image_path)                       # 1.0 = branco (livre), 0.0 = preto (ocupado)
    occ_prob = img if meta.get('negate', 0) else 1.0 - img

    grid = np.full(img.shape, -1, dtype=np.int8)     # -1 = desconhecido
    grid[occ_prob > meta['occupied_thresh']] = 100   # ocupado
    grid[occ_prob < meta['free_thresh']] = 0         # livre

    # No PGM a linha 0 e o topo; no OccupancyGrid a linha 0 esta em y = origin.y
    grid = np.flipud(grid)

    return grid, float(meta['resolution']), [float(v) for v in meta['origin']]


class MapPublisher(Node):
    def __init__(self):
        # Usa sempre o tempo do bag (/clock)
        super().__init__(
            'map_publisher',
            parameter_overrides=[Parameter('use_sim_time', Parameter.Type.BOOL, True)],
        )

        yaml_path = self.declare_parameter('yaml', DEFAULT_YAML).value
        self.frame_id = self.declare_parameter('frame_id', 'odom').value
        publish_tf = self.declare_parameter('publish_tf', False).value
        child_frame = self.declare_parameter('child_frame', 'odom').value
        period = self.declare_parameter('period', 0.1).value

        grid, resolution, origin = load_occupancy(yaml_path)
        height, width = grid.shape

        self.msg = OccupancyGrid()
        self.msg.header.frame_id = self.frame_id
        self.msg.info.resolution = resolution
        self.msg.info.width = width
        self.msg.info.height = height
        self.msg.info.origin.position.x = origin[0]
        self.msg.info.origin.position.y = origin[1]
        yaw = origin[2] if len(origin) > 2 else 0.0
        self.msg.info.origin.orientation.z = float(np.sin(yaw / 2.0))
        self.msg.info.origin.orientation.w = float(np.cos(yaw / 2.0))
        self.msg.data = grid.flatten().tolist()

        # TRANSIENT_LOCAL = "latched": subscritores novos recebem o ultimo mapa
        qos = QoSProfile(
            depth=1,
            history=HistoryPolicy.KEEP_LAST,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.pub = self.create_publisher(OccupancyGrid, '/map', qos)

        if publish_tf and child_frame != self.frame_id:
            self.tf_broadcaster = StaticTransformBroadcaster(self)
            t = TransformStamped()
            t.header.frame_id = self.frame_id
            t.child_frame_id = child_frame
            t.transform.rotation.w = 1.0  # identidade
            self.tf_broadcaster.sendTransform(t)
            self.get_logger().info(f'TF estatica: {self.frame_id} -> {child_frame} (identidade)')

        self.publish_map()
        # Timer em relogio de parede: dispara mesmo que o /clock esteja parado ou volte atras
        self.create_timer(period, self.publish_map, clock=Clock(clock_type=ClockType.STEADY_TIME))

        self.get_logger().info(
            f'/map: {width}x{height} celulas, {resolution} m/celula, origem {origin[:2]}, '
            f'frame "{self.frame_id}", republicado a cada {period}s'
        )

    def publish_map(self):
        self.msg.header.stamp = self.get_clock().now().to_msg()  # tempo do bag
        self.pub.publish(self.msg)


def main():
    rclpy.init()
    node = MapPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
