#!/usr/bin/env python3

from isaacgym import gymapi
from typing import Tuple, Dict, Optional
from tempfile import TemporaryDirectory
from pathlib import Path
import numpy as np
import torch as th
from tqdm.auto import tqdm

from ham.env.episode.spec import DefaultSpec


def _create_table_asset_options(fused: bool = False):
    asset_options = gymapi.AssetOptions()
    # asset_options.fix_base_link = True
    asset_options.fix_base_link = False
    asset_options.flip_visual_attachments = False
    asset_options.vhacd_enabled = False
    asset_options.thickness = 0.001  # ????
    # asset_options.thickness = 0.02  # ????
    asset_options.convex_decomposition_from_submeshes = fused
    # asset_options.convex_decomposition_from_submeshes = True
    asset_options.override_com = False
    asset_options.override_inertia = False
    asset_options.disable_gravity = True
    return asset_options


def _load_table_assets(texts,
                       geoms,
                       gym,
                       sim,
                       as_urdf: bool = False,
                       fuse: bool = False,
                       tmpdir: Optional[str] = None):
    s = geoms.shape
    asset_options = _create_table_asset_options(fuse)
    table_assets = []

    if tmpdir is None:
        tempdir = TemporaryDirectory()
        tmpdirname = tempdir.name
    else:
        tmpdirname = tmpdir

    for i_batch in tqdm(range(texts.shape[0]), desc='env'):
        if fuse:
            # local cache for fused urdf
            filename = (
                Path(tmpdirname) / F'table.urdf'
            )
            if not Path(filename).exists():
                with open(str(filename), 'w') as fp:
                    fp.write(texts[i_batch])
            table_asset = gym.load_urdf(sim, tmpdir,
                                        filename.name,
                                        asset_options)
            table_assets.append(table_asset)
        else:
            table_assets_i = []
            for i_part in range(texts.shape[1]):
                if as_urdf:
                    # == as-urdf ==
                    text = texts[i_batch, i_part]
                    filename = (Path(tmpdir) /
                                F'plate_{i_batch:02d}_{i_part:02d}.urdf')
                    with open(str(filename), 'w') as fp:
                        fp.write(text)
                    # NOTE(ycho): are these options necessary?
                    asset_options.density = 0.0
                    asset_options.override_com = False
                    asset_options.override_inertia = False
                    table_asset = gym.load_urdf(sim, tmpdir,
                                                filename.name,
                                                asset_options)
                else:
                    # == as-box ==
                    # NOTE(ycho): _STRONG_ assumption that
                    # `geoms` are given as axis-aligned boxes
                    # (probably ok for now)
                    asset_options.density = 0.0
                    asset_options.override_com = False
                    asset_options.override_inertia = False
                    x, y, z = [float(x) for x in geoms[i_batch, i_part]]
                    table_asset = gym.create_box(sim, x, y, z,
                                                    asset_options)
                table_assets_i.append(table_asset)
            table_assets.append(table_assets_i)

    if tmpdir is None:
        tempdir.cleanup()

    if fuse:
        table_assets = np.reshape(table_assets, (*s[:-2]))
    else:
        table_assets = np.reshape(table_assets, (*s[:-2], -1))
    return table_assets


class TableAsset(DefaultSpec):
    @property
    def setup_keys(self) -> Tuple[str, ...]: return (
        # FIXME(ycho): should maybe be "plate_geom"
        # to avoid confusion with `table_dim`.
        'table_geom',
        # 'table_urdf',
        # TODO(ycho):
        # think about whether this
        # should be included here...!!
        'table_asset'
    )

    @property
    def setup_deps(self) -> Tuple[str, ...]: return ('table_dim',)

    def sample_setup(self,
                     ctx,
                     data: Dict[str, th.Tensor]) -> Dict[str, th.Tensor]:
        gen = ctx['gen']
        geom = gen.geom(data['table_dim'])
        data['table_geom'] = geom
        fuse_pose = ctx['fuse_pose']
        # Colfree with meshenv requires urdf during setup 
        # have to be updated to avoid saving urdf twice at future
        if fuse_pose:
            tmpdir = ctx['tmpdir']
            outputs = gen.urdf(
                data['table_dim'],
                tmpdir,
                fuse_pose=fuse_pose,
                geom=data['table_geom']
            )
            for i_batch in range(outputs['urdf'].shape[0]):
                filename = (
                    Path(tmpdir) / F'table_{i_batch:02d}.urdf'
                )
                with open(str(filename), 'w') as fp:
                    fp.write(outputs['urdf'][i_batch])
        return data

    def apply_setup(self,
                    ctx,
                    data: Dict[str, th.Tensor]):
        # data = dict(data)

        # == URDF text format ==
        # (gen, tmpdir) -> table_urdf_text
        gen = ctx['gen']
        gym = ctx.get('gym', None)
        tmpdir = ctx['tmpdir']
        fuse_pose = ctx['fuse_pose']
        outputs = gen.urdf(
            data['table_dim'],
            tmpdir,
            fuse_pose=fuse_pose,
            geom=data['table_geom']
        )
        # data['table_urdf_text'] = outputs['urdf']

        # == as Isaac Gym asset ==
        # (table_urdf_text, table_geom) -> table_asset
        # NOTE(ycho): we allow gym=None for mock testing
        if gym is not None:
            assets = _load_table_assets(
                # data['table_urdf_text'],
                outputs['urdf'],
                data['table_geom'],
                gym,
                ctx['sim'],
                False,
                fuse_pose,
                tmpdir if fuse_pose else None)
            # NOTE(ycho): list -> dict
            data['table_asset'] = {i: v for (i, v) in enumerate(assets)}
        return data


def main():
    table_asset = TableAsset()


if __name__ == '__main__':
    main()
