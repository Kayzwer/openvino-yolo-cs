namespace yolov8_openvino
{
    public class YoloClassifyPrediction(string className, int classIndex, float score)
    {
        public string ClassName { get; set; } = className;
        public int ClassIndex { get; set; } = classIndex;
        public float Score { get; set; } = score;
    }
}
