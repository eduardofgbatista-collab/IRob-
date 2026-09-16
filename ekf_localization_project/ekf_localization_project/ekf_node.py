import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped
from visualization_msgs.msg import Marker
from tf2_ros import Buffer, TransformException, TransformListener
import numpy as np
import math

class EKFNode(Node):
    def __init__(self):
        super().__init__('ekf_node')
        
        self.state = np.zeros((3, 1))
        self.P = np.eye(3) * 0.1 
        self.Q = np.diag([0.01, 0.01, 0.01])
        
        # Matriz de Ruído da Medição (R) - O Ground Truth é muito preciso, logo o ruído é baixo
        self.R = np.diag([0.001, 0.001, 0.001]) 
        
        self.last_time = None
        self.is_initialized = False
        self.last_gt_stamp = None

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        
        # Subscrições
        self.odom_sub = self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        
        # The dataset publishes ground truth as mocap -> mocap_laser_link on /tf.
        self.gt_timer = self.create_timer(1.0 / 60.0, self.gt_callback)
        
        # Publicadores
        self.pose_pub = self.create_publisher(PoseStamped, '/pose', 10)
        self.marker_pub = self.create_publisher(Marker, '/ekf_point', 10)
        
        self.get_logger().info("Nó EKF (Previsão + Correção) iniciado!")

    def gt_callback(self):
        if not self.is_initialized:
            return

        try:
            transform = self.tf_buffer.lookup_transform(
                'odom', 'mocap_laser_link', rclpy.time.Time()
            )
        except TransformException:
            return

        stamp = transform.header.stamp.sec * 10**9 + transform.header.stamp.nanosec
        if stamp == self.last_gt_stamp:
            return
        self.last_gt_stamp = stamp

        # Extrair a medição real (z)
        z = np.zeros((3, 1))
        z[0, 0] = transform.transform.translation.x
        z[1, 0] = transform.transform.translation.y
        
        q = transform.transform.rotation
        siny_cosp = 2 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
        z[2, 0] = math.atan2(siny_cosp, cosy_cosp)
        
        self.correction_step(z)

    def correction_step(self, z):
        # Matriz de Observação H (Observamos x, y, theta diretamente)
        H = np.eye(3)
        
        # 1. Inovação (y = z - H*x)
        y = z - H @ self.state
        y[2, 0] = math.atan2(math.sin(y[2, 0]), math.cos(y[2, 0])) # Normalizar ângulo
        
        # 2. Covariância da Inovação (S = H*P*H^T + R)
        S = H @ self.P @ H.T + self.R
        
        # 3. Ganho de Kalman (K = P*H^T*S^-1)
        K = self.P @ H.T @ np.linalg.inv(S)
        
        # 4. Atualizar o Estado
        self.state = self.state + K @ y
        self.state[2, 0] = math.atan2(math.sin(self.state[2, 0]), math.cos(self.state[2, 0]))
        
        # 5. Atualizar a Covariância (P = (I - K*H)*P)
        I = np.eye(3)
        self.P = (I - K @ H) @ self.P
        
        self.get_logger().info("EKF: Correção aplicada!")

    def odom_callback(self, msg):
        current_time = self.get_clock().now().nanoseconds / 1e9
        
        if not self.is_initialized:
            self.state[0, 0] = msg.pose.pose.position.x
            self.state[1, 0] = msg.pose.pose.position.y
            q = msg.pose.pose.orientation
            siny_cosp = 2 * (q.w * q.z + q.x * q.y)
            cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
            self.state[2, 0] = math.atan2(siny_cosp, cosy_cosp)
            self.last_time = current_time
            self.is_initialized = True
            return
            
        dt = current_time - self.last_time
        self.last_time = current_time
        
        if dt > 1.0 or dt < 0.0:
            return
            
        v = msg.twist.twist.linear.x
        omega = msg.twist.twist.angular.z
        
        self.prediction_step(v, omega, dt)
        self.publish_ekf_state()

    def prediction_step(self, v, omega, dt):
        x = self.state[0, 0]
        y = self.state[1, 0]
        theta = self.state[2, 0]
        
        self.state[0, 0] = x + v * dt * math.cos(theta)
        self.state[1, 0] = y + v * dt * math.sin(theta)
        self.state[2, 0] = theta + omega * dt
        self.state[2, 0] = math.atan2(math.sin(self.state[2, 0]), math.cos(self.state[2, 0]))
        
        A = np.array([
            [1.0, 0.0, -v * dt * math.sin(theta)],
            [0.0, 1.0,  v * dt * math.cos(theta)],
            [0.0, 0.0, 1.0]
        ])
        self.P = A @ self.P @ A.T + self.Q

    def publish_ekf_state(self):
        pose = PoseStamped()
        pose.header.stamp = self.get_clock().now().to_msg()
        pose.header.frame_id = 'odom'
        pose.pose.position.x = float(self.state[0, 0])
        pose.pose.position.y = float(self.state[1, 0])
        theta = float(self.state[2, 0])
        pose.pose.orientation.z = math.sin(theta / 2.0)
        pose.pose.orientation.w = math.cos(theta / 2.0)
        self.pose_pub.publish(pose)

        # Ponto Verde
        marker = Marker()
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.header.frame_id = 'odom'
        marker.ns = 'ekf_prediction'
        marker.id = 0
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.pose.position.x = float(self.state[0, 0])
        marker.pose.position.y = float(self.state[1, 0])
        marker.pose.position.z = 0.1
        marker.pose.orientation.w = 1.0
        marker.scale.x = 0.1
        marker.scale.y = 0.1
        marker.scale.z = 0.1
        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 0.0
        marker.color.a = 1.0
        self.marker_pub.publish(marker)

def main(args=None):
    rclpy.init(args=args)
    node = EKFNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()