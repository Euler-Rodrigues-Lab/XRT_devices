"""TCP acquisition works with only head and hand observations."""
from types import SimpleNamespace as NS
import numpy as np
from xrt_devices.integrations.tcp import bones_to_tcp_action,mediapipe_to_tcp_action
from xrt_devices.schemas.openxr_skeletons import FullBodyBoneId as B
from xrt_devices.schemas.body_pose import Bone


def test_xr_requires_no_torso_or_elbows():
    rng=np.random.default_rng(15)
    bones=[Bone(int(b),rng.normal(size=3),(0,0,0,1)) for b in B
           if b==B.FullBody_Head or 'Hand' in b.name]
    action=bones_to_tcp_action(bones)
    assert action['engaged']
    for side in ('left','right'):
        r=action[side+'_eef']['rotation']
        np.testing.assert_allclose(r.T@r,np.eye(3),atol=1e-12)
        assert len(action[side+'_fingers_wrist'])==5
    assert bones_to_tcp_action([b for b in bones if b.id!=B.FullBody_Head]) is None
    assert not any('sew' in k or 'body' in k for k in action)


def test_mediapipe_ignores_all_torso_and_elbow_points():
    point=lambda p:NS(x=p[0],y=p[1],z=p[2],visibility=1.)
    world=[point([np.nan]*3) for _ in range(33)]
    screen=[point([np.nan]*3) for _ in range(33)]
    for i,p in {0:[.1,0,1.],7:[0,.1,1.],8:[0,-.1,1.],15:[.4,.3,.6],16:[.4,-.3,.6]}.items():
        world[i]=point(p);screen[i]=point(p)
    pose=NS(pose_world_landmarks=NS(landmark=world),pose_landmarks=NS(landmark=screen))
    hands=NS(multi_hand_landmarks=[],multi_hand_world_landmarks=[])
    for wi in [16,15]:
        local=np.random.default_rng(wi).normal(size=(21,3))*.03
        hands.multi_hand_landmarks.append(NS(landmark=[screen[wi]]*21))
        hands.multi_hand_world_landmarks.append(NS(landmark=[point(p) for p in local]))
    action=mediapipe_to_tcp_action(pose,hands,np.eye(3))
    assert action['engaged']
    for side,wi in [('left',15),('right',16)]:
        np.testing.assert_allclose(action[side+'_eef']['position'],[world[wi].x,world[wi].y,world[wi].z])
        assert len(action[side+'_fingers_wrist'])==5
    np.testing.assert_allclose(action['head_pose']['rotation'],np.eye(3),atol=1e-12)
