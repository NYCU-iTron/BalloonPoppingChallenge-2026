import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

_state = {"fig": None, "ax": None, "art": None, "trail": [], "view": None}


def _quat_rotate(quat, vec):
    qw, qx, qy, qz = quat
    qv = np.array([qx, qy, qz])
    t = 2.0 * np.cross(qv, vec)
    return vec + qw * t + np.cross(qv, t)


def reset_scene():
    _state["trail"] = []
    _state["view"] = None


def _init_artists(ax):
    art = {}
    art["rocket"] = ax.scatter([], [], [], c="blue", marker="^", s=90,
                               depthshade=False)
    art["target"] = ax.scatter([], [], [], facecolors="none", edgecolors="red",
                               marker="o", s=200, linewidths=2.0, depthshade=False)
    (art["los"],)   = ax.plot([], [], [], color="gold", ls="--", lw=1.5)
    (art["trail"],) = ax.plot([], [], [], color="navy", lw=1.0, alpha=0.5)
    (art["nose"],)  = ax.plot([], [], [], color="blue", lw=2.0)
    art["txt"] = ax.text2D(0.02, 0.98, "", transform=ax.transAxes, va="top", fontsize=9)
    ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)"); ax.set_zlabel("Z (m)")
    ax.set_box_aspect((1, 1, 1))          # 立方體比例:角度/距離不失真
    legend_handles = [
        Line2D([0], [0], color="none", marker="^", markerfacecolor="blue",
               markeredgecolor="blue", markersize=9, label="Rocket"),
        Line2D([0], [0], color="none", marker="o", markerfacecolor="none",
               markeredgecolor="red", markeredgewidth=2, markersize=10,
               label="Target"),
        Line2D([0], [0], color="gold", linestyle="--", linewidth=1.5,
               label="Line of sight"),
        Line2D([0], [0], color="navy", linewidth=1.0, alpha=0.5,
               label="Rocket trail"),
        Line2D([0], [0], color="blue", linewidth=2.0, label="Nose direction"),
    ]
    ax.legend(
        handles=legend_handles,
        loc="lower left",
        bbox_to_anchor=(1.02, 0.02),
        borderaxespad=0.0,
        fontsize=8,
        frameon=True,
    )
    return art


def _set3d(scat, pt):
    scat._offsets3d = ([], [], []) if pt is None else ([pt[0]], [pt[1]], [pt[2]])


def _update_view(ax, focus, margin, min_half):
    """只框住 focus(火箭+目標),維持立方體;有遲滯,只在內容跑出框時才重設。"""
    lo, hi = focus.min(axis=0), focus.max(axis=0)
    c = (lo + hi) / 2.0
    half = max((hi - lo).max() / 2.0 + margin, min_half)   # 立方體半邊長,有下限

    v = _state["view"]
    if v is not None:
        cc, ch = v
        inside  = np.all(np.abs(focus - cc) <= ch)         # 內容還在現有框內?
        too_big = ch > half * 2.0                          # 框比需要大太多?
        if inside and not too_big:
            return                                         # 不動 → 最快路徑,不觸發 3D 重畫

    _state["view"] = (c, half)
    ax.set_xlim(c[0] - half, c[0] + half)
    ax.set_ylim(c[1] - half, c[1] + half)
    ax.set_zlim(c[2] - half, c[2] + half)


def render_scene(observation, rocket_state, balloon_state=None,
                 show=True, arrow_length=12.0, trail_length=400,
                 margin=20.0, min_half=25.0):
    if _state["fig"] is None or not plt.fignum_exists(_state["fig"].number):
        fig = plt.figure(figsize=(8.8, 7))
        fig.subplots_adjust(right=0.78)
        ax = fig.add_subplot(projection="3d")
        _state.update(fig=fig, ax=ax, art=_init_artists(ax), view=None)
        if show:
            plt.show(block=False)
    ax, art = _state["ax"], _state["art"]

    rs = np.asarray(rocket_state, dtype=float)
    rpos, rquat = rs[:3], rs[6:10]
    rvalid = np.isfinite(rpos).all()

    tpos = None
    if balloon_state is not None:
        bs = np.asarray(balloon_state, dtype=float)
        if np.isfinite(bs[:3]).all():
            tpos = bs[:3]

    # 火箭 + 軌跡 + 機頭
    if rvalid:
        _set3d(art["rocket"], rpos)
        _state["trail"].append(rpos.copy())
        if len(_state["trail"]) > trail_length:
            _state["trail"].pop(0)
        tr = np.asarray(_state["trail"])
        art["trail"].set_data_3d(tr[:, 0], tr[:, 1], tr[:, 2])
        if np.isfinite(rquat).all() and np.linalg.norm(rquat) > 1e-6:
            nose = _quat_rotate(rquat, np.array([0.0, 0.0, 1.0]))
            nose = nose / (np.linalg.norm(nose) + 1e-12) * arrow_length
            tip = rpos + nose
            art["nose"].set_data_3d([rpos[0], tip[0]], [rpos[1], tip[1]], [rpos[2], tip[2]])
    else:
        _set3d(art["rocket"], None)

    # 目標 + LOS + 距離文字
    if tpos is not None:
        _set3d(art["target"], tpos)
        if rvalid:
            art["los"].set_data_3d([rpos[0], tpos[0]], [rpos[1], tpos[1]], [rpos[2], tpos[2]])
            art["txt"].set_text(f"dist {np.linalg.norm(tpos - rpos):6.1f} m")
        else:
            art["los"].set_data_3d([], [], [])
    else:
        _set3d(art["target"], None)
        art["los"].set_data_3d([], [], [])
        art["txt"].set_text("no target")

    # 顯示範圍最佳化:只看 火箭+目標
    focus = [p for p in (rpos if rvalid else None, tpos) if p is not None]
    if focus:
        _update_view(ax, np.array(focus), margin, min_half)

    if show:
        _state["fig"].canvas.draw_idle()
        _state["fig"].canvas.flush_events()
    return ax
