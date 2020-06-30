bazel build -c opt --copt -DMESA_EGL_NO_X11_HEADERS \
    mediapipe/examples/desktop/hand_tracking:hand_tracking_gpu


GLOG_logtostderr=1 bazel-bin/mediapipe/examples/desktop/hand_tracking/hand_tracking_gpu \
    --calculator_graph_config_file=mediapipe/graphs/hand_tracking/hand_tracking_mobile.pbtxt
