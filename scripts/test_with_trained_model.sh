## test with trained model
# perform reconstruction and evaluate geometry
python tools/test_net.py -c configs/cat/joint_stage2.yaml solver.load latest solver.load_model "" recon True
# perform rendering and evaluate mask
python tools/test_net.py -c configs/cat/joint_stage2.yaml solver.load latest solver.load_model "" dont_render_bg True tf 1 model.pokingrecon.mdepth_fill True

# visualize
wis3d --vis_dir dbg --host localhost