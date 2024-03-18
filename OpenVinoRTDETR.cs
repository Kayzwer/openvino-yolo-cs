using OpenCvSharp.Dnn;
using OpenCvSharp;
using Sdcb.OpenVINO;
using System.Drawing;
using System.Xml.Linq;
using System.Xml.XPath;
using Sdcb.OpenVINO.Extensions.OpenCvSharp4;

namespace OpenVinoYOLO
{
    public class OpenVinoRTDETR
    {
        string[] classes { get; set; }
        Model model { get; set; }
        PrePostProcessor prePostProcessor { get; set; }
        PreProcessInputInfo preProcessInputInfo { get; set; }
        CompiledModel compiledModel { get; set; }
        InferRequest inferRequest { get; set; }
        Shape inputShape { get; set; }
        double _255_inv = 1.0 / 255.0;
        Dictionary<string, Color> colorMapper { get; set; }
        int rowCount { get; set; }
        Dictionary<int, int> rowCaches { get; set; }
        int objectCount { get; set; }

        public OpenVinoRTDETR(string model_xml_path, bool use_gpu)
        {
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
            rowCount = classes.Length + 4;
            colorMapper = [];
            objectCount = (int)model.Outputs.Primary.Shape.ElementCount / rowCount;
            rowCaches = [];
            for (int i = 0; i < objectCount; i++)
            {
                rowCaches.Add(i, i * rowCount);
            }
        }

        public void SetupColors(Color[] colors)
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
            using Mat F32 = new();
            resized.ConvertTo(F32, MatType.CV_32FC3, _255_inv);
            using Tensor input = F32.AsTensor();
            inferRequest.Inputs.Primary = input;
            inferRequest.Run();
            using Tensor output = inferRequest.Outputs.Primary;
            float[] data = output.GetData<float>().ToArray();
            List<YoloPrediction> predictions = [];
            for (int i = 0; i < objectCount; i++)
            {
                int rowCache = rowCaches[i];
                float[] confs = data[(rowCache + 4)..(rowCache + rowCount)];
                int maxConfIndex = IndexOfMax(confs);
                float conf = confs[maxConfIndex];
                float[] rectData = data[rowCache..(rowCache + 4)];
                float x = rectData[0] * image.Width;
                float y = rectData[1] * image.Height;
                float w = rectData[2] * image.Width;
                float h = rectData[3] * image.Height;
                predictions.Add(
                    new(
                        new(maxConfIndex, classes[maxConfIndex], colorMapper[classes[maxConfIndex]]),
                        new(x - w * .5f, y - h * .5f, w, h), conf)
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
    }
}
