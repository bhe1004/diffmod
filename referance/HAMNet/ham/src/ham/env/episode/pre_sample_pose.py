#!/usr/bin/env python3

from typing import Tuple, Dict
import torch as th

from ham.env.episode.spec import DefaultSpec
from ham.env.episode.util import upsert
from ham.util.math_util import quat_multiply
from ham.env.scene.sample_pose import (
    rejection_sample
)
from ham.env.task.reward_util import (
    pose_error
)
from ham.util.torch_util import randu
class GetObjectOrn(DefaultSpec):

    @property
    def reset_keys(self) -> Tuple[str, ...]: return (
        'obj_orn',
        'which_pose',
        'goal_index'
    )

    @property
    def reset_deps(self) -> Tuple[str, ...]: return (
        'reset_ids',
        'obj_stable_pose',
)

    def sample_reset(
            self, ctx, data: Dict[str, th.Tensor]) -> Dict[str, th.Tensor]:
        gen = ctx['gen']
        reset_ids = data['reset_ids']
        num_reset: int = len(reset_ids)
        if len(reset_ids) <= 0:
            return data
        stable_poses = data['obj_stable_pose']
        indices = th.multinomial(th.ones(num_reset, stable_poses.shape[-2],
                                         device=stable_poses.device),
                                  2,
                                  replacement=False
                                  ) # first for init and second for goal
        which_pose = indices[..., 0:1]
        qs = th.take_along_dim(stable_poses[reset_ids, ..., 3:7],
                               which_pose[..., None],
                               -2).squeeze(-2)
        goal_index = indices[..., 1:2]

        upsert(data, reset_ids, 'obj_orn', qs)
        upsert(data, reset_ids, 'which_pose', which_pose)
        upsert(data, reset_ids, 'goal_index', goal_index)
        return data

class UsePresampledPose(DefaultSpec):

    @property
    def reset_keys(self) -> Tuple[str, ...]: return (
        'obj_poses',
        'goal_radius',
        'goal_angle',
        'goal_poses',
    )

    @property
    def reset_deps(self) -> Tuple[str, ...]: return (
        'reset_ids',
        'curr_pose_meta',
        'table_dim',
        'table_pos',
        'obj_radius',
        'obj_stable_pose',
        'obj_hull',
        'which_pose',
        'goal_index',
        'obj_orn',
    )

    def sample_reset(
            self, ctx, data: Dict[str, th.Tensor]) -> Dict[str, th.Tensor]:
        reset_ids = data['reset_ids']

        num_reset: int = len(reset_ids)
        if num_reset <= 0:
            return data
        
        stable_poses = data['obj_stable_pose']
        which_pose = data['which_pose'][reset_ids]
        xyz = th.take_along_dim(stable_poses[reset_ids, ..., :3],
                               which_pose[..., None],
                               -2).squeeze(-2)
        quat = data['obj_orn'][reset_ids]
        table_pos = data['table_pos'][reset_ids]
        meta = data['curr_pose_meta']['pose']
        xyz[..., 2] += meta[..., 0, 2]
        xyz[..., :2] += table_pos[..., :2]
        xyz[..., 2] += ctx['z_eps']

        init_pose = th.cat([xyz, quat], dim=-1)

        upsert(data, reset_ids, 'obj_poses', init_pose[..., None, :])

        r_lo, r_hi = ctx['goal_radius_bound']
        a_lo, a_hi = ctx['goal_angle_bound']
        new_goal_radius = randu(r_lo, r_hi,
                                (num_reset,),
                                device=ctx['device'])
        new_goal_angle = randu(a_lo, a_hi,
                               (num_reset,),
                               device=ctx['device'])
        upsert(data, reset_ids,
               'goal_radius',
               new_goal_radius)
        upsert(data, reset_ids,
               'goal_angle',
               new_goal_angle)
        
       
        count = min(stable_poses.shape[1], 16)
        n_env = len(reset_ids)
        def sample_fn():
            # == sample pose ==
            goal_index = th.randint(
                            stable_poses.shape[1],
                            size=(n_env, count),
                            dtype=th.long,
                            device=stable_poses.device,
                        )
            xyz = th.take_along_dim(stable_poses[reset_ids, ..., :3],
                            goal_index[..., None],
                            -2).squeeze(-2)
            quat = th.take_along_dim(stable_poses[reset_ids, ..., 3:7],
                            goal_index[..., None],
                            -2).squeeze(-2)
            return th.cat([xyz, quat], dim=-1).swapaxes(0,1)
        
        def accept_fn(pose):
            src_pose = init_pose[..., None, :].swapaxes(0,-2).expand(
                (*pose.shape[:-1], 7))
            
            pos_err, orn_err = pose_error(
                pose[..., 0:3], src_pose[..., 0:3],
                pose[..., 3:7], src_pose[..., 3:7]
            )
            
            return th.logical_or(
                pos_err > data['goal_radius'][reset_ids][None],
                orn_err > data['goal_angle'][reset_ids][None],
            )
        goal_pose = rejection_sample(sample_fn,
                                    accept_fn,
                                    batched=True,
                                    sample=True,
                                    )
 
        goal_pose[..., 2] += meta[..., 0, 2]
       
        goal_pose[..., :2] += table_pos[..., :2]
        upsert(data, reset_ids, 'goal_poses', goal_pose[..., None, :])

        return data

    def apply_reset(self,
                    ctx,
                    data: Dict[str, th.Tensor]):
       reset_ids = data['reset_ids']
       if len(reset_ids) <= 0:
           return data
       set_ids = data['obj_id'][reset_ids]
       root_tensor = ctx['root_tensor']
       root_tensor[set_ids, :7] = data['obj_pose'][reset_ids]
       return data