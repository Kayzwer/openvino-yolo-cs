using System.Drawing;

namespace yolov8_openvino
{
    public class OBBPrediction(YoloLabel label, RectangleF rectangle, float angle, float confidence)
    {
        public YoloLabel Label { get; set; } = label;
        public RectangleF Rectangle { get; set; } = rectangle;
        public float Area { get; set; } = rectangle.Area();
        public float Score { get; set; } = confidence;
        public float Angle { get; set; } = angle;
    }
}
