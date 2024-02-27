using OpenCvSharp;
using OpenCvSharp.Dnn;
using Sdcb.OpenVINO;
using Sdcb.OpenVINO.Extensions.OpenCvSharp4;
using System.Drawing;
using System.Xml.Linq;
using System.Xml.XPath;

namespace yolov8_openvino
{
    public class OpenVinoYolov8
    {
        string model_xml_path { get; set; }
        string[] classes { get; set; }
        Model model { get; set; }
        PrePostProcessor prePostProcessor { get; set; }
        PreProcessInputInfo preProcessInputInfo { get; set; }
        CompiledModel compiledModel { get; set; }
        InferRequest inferRequest { get; set; }
        Shape inputShape { get; set; }
        double inputWidthInv { get; set; }
        double inputHeightInv { get; set; }
        double _255_inv = 1.0 / 255.0;
        Dictionary<string, Color> colorMapper { get; set; }
        int rowCount { get; set; }
        Dictionary<int, int> rowCaches { get; set; }
        int objectCount { get; set; }

        public OpenVinoYolov8(string model_xml_path, bool use_gpu)
        {
            this.model_xml_path = model_xml_path;
            classes = XDocument.Load(model_xml_path).XPathSelectElement(@"/net/rt_info/model_info/labels")!.Attribute("value")!.Value.Split(" ");
            model = OVCore.Shared.ReadModel(model_xml_path);
            prePostProcessor = model.CreatePrePostProcessor();
            preProcessInputInfo = prePostProcessor.Inputs.Primary;
            preProcessInputInfo.TensorInfo.Layout = Layout.NHWC;
            preProcessInputInfo.ModelInfo.Layout = Layout.NCHW;
            model = prePostProcessor.BuildModel();
            compiledModel = OVCore.Shared.CompileModel(model, use_gpu ? "GPU" : "CPU");
            inferRequest = compiledModel.CreateInferRequest();
            inputShape = model.Inputs.Primary.Shape;
            inputWidthInv = 1.0 / inputShape[2];
            inputHeightInv = 1.0 / inputShape[1];
            rowCount = classes.Length + 4;
            colorMapper = [];
            rowCaches = [];
            objectCount = (int)model.Outputs.Primary.Shape.ElementCount / rowCount;
            int accumulation = 0;
            for (int i = 0; i < objectCount; i++)
            {
                rowCaches.Add(i, accumulation);
                accumulation += rowCount;
            }
        }

        public void SetupColor(Color[] colors)
        {
            if (colors.Length != classes.Length)
            {
                throw new Exception("Check number of classes");
            }
            for (int i = 0; i < colors.Length; i++)
            {
                colorMapper.Add(classes[i], colors[i]);
            }
        }

        public List<YoloPrediction> Predict(Mat image, float conf_threshold, float iou_threshold)
        {
            using Mat resized = image.Resize(new OpenCvSharp.Size(inputShape[2], inputShape[1]));
            Size2f sizeRatio = new(image.Width * inputWidthInv, image.Height * inputHeightInv);
            using Mat F32 = new();
            resized.ConvertTo(F32, MatType.CV_32FC3, _255_inv);
            using Tensor input = F32.AsTensor();
            inferRequest.Inputs.Primary = input;
            inferRequest.Run();
            using Tensor output = inferRequest.Outputs.Primary;
            Span<float> data = output.GetData<float>();
            float[] t = Transpose(data, output.Shape[1], output.Shape[2]);
            List<YoloPrediction> predictions = [];
            for (int i = 0; i < objectCount; i++)
            {
                int rowCache = rowCaches[i];
                Span<float> confs = t.AsSpan()[(rowCache + 4)..(rowCache + rowCount)];
                int maxConfIndex = IndexOfMax(confs);
                float conf = confs[maxConfIndex];
                Span<float> rectData = t.AsSpan()[rowCache..(rowCache + 4)];
                float x = rectData[0] * sizeRatio.Width;
                float y = rectData[1] * sizeRatio.Height;
                float w = rectData[2] * sizeRatio.Width;
                float h = rectData[3] * sizeRatio.Height;
                predictions.Add(
                    new(
                        new(maxConfIndex, classes[maxConfIndex], colorMapper[classes[maxConfIndex]]),
                        new(x - w * .5f, y - h * .5f, w, h), conf
                        )
                    );
            }
            CvDnn.NMSBoxes(
                predictions.Select(x => x.Rectangle.ToRect()),
                predictions.Select(x => x.Score),
                conf_threshold,
                iou_threshold,
                out int[] indices);
            return predictions.Where((x, i) => indices.Contains(i)).ToList();
        }

        static int IndexOfMax(ReadOnlySpan<float> data)
        {
            if (data.Length == 0) throw new ArgumentException("The provided data span is null or empty.");
            int maxIndex = 0;
            float maxValue = data[0];
            for (int i = 1; i < data.Length; i++)
            {
                if (data[i] > maxValue)
                {
                    maxValue = data[i];
                    maxIndex = i;
                }
                if (maxValue > .5f)
                {
                    break;
                }
            }
            return maxIndex;
        }

        static unsafe float[] Transpose(ReadOnlySpan<float> tensorData, int rows, int cols)
        {
            float[] transposedTensorData = new float[tensorData.Length];
            fixed (float* pTensorData = tensorData)
            {
                fixed (float* pTransposedData = transposedTensorData)
                {
                    for (int i = 0; i < rows; i++)
                    {
                        int colCache = i * cols;
                        for (int j = 0; j < cols; j++)
                        {
                            int index = colCache + j;
                            int transposedIndex = j * rows + i;
                            pTransposedData[transposedIndex] = pTensorData[index];
                        }
                    }
                }
            }
            return transposedTensorData;
        }
    }
}
