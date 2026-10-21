"""Head/wrist/hand-only tracking. No shoulder, torso or elbow observations read."""
from itertools import permutations
import numpy as np
from xrt_devices.processing import _get_hand_frame, q2R
from xrt_devices.schemas.openxr_skeletons import FullBodyBoneId as B
from .eef import complete_hand_landmarks


def _hands(positions):
    action={}
    for side in ('left','right'):
        required=[getattr(B,f'FullBody_{side.title()}Hand{k}') for k in ('Wrist','IndexProximal','LittleProximal')]
        if any(k not in positions for k in required):continue
        r=_get_hand_frame(side,positions)
        if r is None or not np.isfinite(r).all():continue
        wrist=positions[required[0]]
        named={B(k).name:v for k,v in positions.items()}
        action[f'{side}_eef']={'position':wrist,'rotation':r}
        action[f'{side}_fingers_wrist']=complete_hand_landmarks(named,side,wrist,r)
    return action


def bones_to_tcp_action(bones,tags=None):
    bones=list(bones)
    positions={b.id:np.asarray(b.position,float) for b in bones
               if (b.id==B.FullBody_Head or 'Hand' in B(b.id).name) and np.isfinite(b.position).all()}
    head=next((b for b in bones if b.id==B.FullBody_Head),None)
    if head is None or head.id not in positions:return None
    q=np.asarray(head.rotation,float)
    if q.shape!=(4,) or not np.isfinite(q).all() or np.linalg.norm(q)<1e-8:return None
    # Same OpenXR-head to x-forward/y-left/z-up conversion as the shared device.
    correction=np.array([[0.,1.,0.],[0.,0.,1.],[1.,0.,0.]])
    action=_hands(positions)
    action['head_pose']={'position':positions[head.id],'rotation':q2R(q/np.linalg.norm(q))@correction}
    action['engaged']=bool(any(f'{s}_eef' in action for s in ('left','right')))
    return action


def mediapipe_to_tcp_action(pose,hands,world_rotation):
    """Metric wrists and face axes from only face/wrist pose points + hand points."""
    if not pose.pose_world_landmarks or not pose.pose_landmarks:return None
    world=pose.pose_world_landmarks.landmark;screen=pose.pose_landmarks.landmark
    point=lambda i:world_rotation@np.array([world[i].x,world[i].y,world[i].z])
    if any(screen[i].visibility<.75 for i in (0,7,8)):return None
    left,right,nose=point(7),point(8),point(0)
    y=left-right
    if np.linalg.norm(y)<1e-8:return None
    y/=np.linalg.norm(y);x=nose-(left+right)/2;x-=y*np.dot(x,y)
    if np.linalg.norm(x)<1e-8:return None
    x/=np.linalg.norm(x);z=np.cross(x,y)
    action={'head_pose':{'position':(left+right)/2,'rotation':np.column_stack((x,y,z))},'engaged':False}
    if not hands or not hands.multi_hand_world_landmarks:return action
    records=list(zip(hands.multi_hand_landmarks,hands.multi_hand_world_landmarks))[:2]
    wrists=[np.array([screen[i].x,screen[i].y]) for i in (15,16)]
    assignment=min(permutations(range(2),len(records)),key=lambda a:sum(np.linalg.norm(np.array([r[0].landmark[0].x,r[0].landmark[0].y])-wrists[a[i]])**2 for i,r in enumerate(records)))
    positions={}
    names=['Wrist','ThumbMetacarpal','ThumbProximal','ThumbDistal','ThumbTip','IndexProximal','IndexIntermediate','IndexDistal','IndexTip','MiddleProximal','MiddleIntermediate','MiddleDistal','MiddleTip','RingProximal','RingIntermediate','RingDistal','RingTip','LittleProximal','LittleIntermediate','LittleDistal','LittleTip']
    for i,(_,world_hand) in enumerate(records):
        side=('Left','Right')[assignment[i]];wi=(15,16)[assignment[i]]
        if screen[wi].visibility<.75:continue
        points=np.array([[p.x,p.y,p.z] for p in world_hand.landmark]);points=(points-points[0])@world_rotation.T+point(wi)
        if not np.isfinite(points).all():continue
        positions.update({getattr(B,f'FullBody_{side}Hand{name}'):p for name,p in zip(names,points)})
    action.update(_hands(positions));action['engaged']=any(f'{s}_eef' in action for s in ('left','right'))
    return action
