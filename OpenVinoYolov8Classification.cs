using OpenCvSharp;
using Sdcb.OpenVINO;
using Sdcb.OpenVINO.Extensions.OpenCvSharp4;
using System.Xml.Linq;
using System.Xml.XPath;

namespace OpenVinoYOLO
{
    public class OpenVinoYolov8Classification
    {
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

        public OpenVinoYolov8Classification(string model_xml_path, bool use_gpu)
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
            inputWidthInv = 1.0 / inputShape[2];
            inputHeightInv = 1.0 / inputShape[1];
        }

        public YoloClassifyPrediction Predict(Mat image)
        {
            using Mat resized = image.Resize(new OpenCvSharp.Size(inputShape[2], inputShape[1]));
            Size2f sizeRatio = new(image.Width * inputWidthInv, image.Height * inputHeightInv);
            using Mat F32 = new();
            resized.ConvertTo(F32, MatType.CV_32FC3, _255_inv);
            using Tensor input = F32.AsTensor();
            inferRequest.Inputs.Primary = input;
            inferRequest.Run();
            using Tensor output = inferRequest.Outputs.Primary;
            return IndexOfMax(output.GetData<float>());
        }

        public YoloClassifyPrediction IndexOfMax(ReadOnlySpan<float> data)
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
            return new(classes[maxIndex], maxIndex, maxValue);
        }
    }
}
