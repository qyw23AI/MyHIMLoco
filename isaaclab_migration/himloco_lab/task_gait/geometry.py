"""Nominal foot centres from URDF forward kinematics, independent of foot order."""
import xml.etree.ElementTree as ET
import numpy as np


def nominal_feet(path, angles, foot_names):
    root = ET.parse(path).getroot()
    transforms = {}
    joints = list(root.findall('joint'))
    children = {j.find('child').get('link') for j in joints}
    base = next(l.get('name') for l in root.findall('link') if l.get('name') not in children)
    transforms[base] = np.eye(4)
    def rotation(axis, angle):
        axis = axis / np.linalg.norm(axis)
        x, y, z = axis
        skew = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
        return np.eye(3) + np.sin(angle) * skew + (1 - np.cos(angle)) * skew @ skew
    while joints:
        progressed = False
        for joint in joints[:]:
            parent = joint.find('parent').get('link')
            if parent not in transforms:
                continue
            origin = joint.find('origin')
            xyz = np.fromstring(origin.get('xyz', '0 0 0'), sep=' ') if origin is not None else np.zeros(3)
            rpy = np.fromstring(origin.get('rpy', '0 0 0'), sep=' ') if origin is not None else np.zeros(3)
            local = np.eye(4)
            local[:3, 3] = xyz
            local[:3, :3] = rotation(np.array([0., 0., 1.]), rpy[2]) @ rotation(np.array([0., 1., 0.]), rpy[1]) @ rotation(np.array([1., 0., 0.]), rpy[0])
            if joint.get('type') != 'fixed':
                if joint.get('type') not in ('revolute', 'continuous'):
                    raise ValueError('Task gait currently supports revolute URDF joints only')
                axis = joint.find('axis')
                vector = np.fromstring(axis.get('xyz', '1 0 0'), sep=' ') if axis is not None else np.array([1., 0., 0.])
                local[:3, :3] = local[:3, :3] @ rotation(vector, angles[joint.get('name')])
            transforms[joint.find('child').get('link')] = transforms[parent] @ local
            joints.remove(joint)
            progressed = True
        if not progressed:
            raise ValueError('URDF joint graph is disconnected')
    return np.stack([transforms[name][:3, 3] for name in foot_names])
