using OpenCvSharp;
using System.Drawing;

namespace OpenVinoYOLO
{
    public static class Extensions
    {
        public static Rect ToRect(this RectangleF rect) => new((int)rect.X, (int)rect.Y, (int)rect.Width, (int)rect.Height);
        public static Image DrawBoundingBox(Image image, List<YoloPrediction> predictions, int bounding_box_thickness, int font_size)
        {
            using Graphics graphics = Graphics.FromImage(image);
            for (int i = 0; i < predictions.Count; i++)
            {
                float score = (float)Math.Round(predictions[i].Score, 2);
                graphics.DrawRectangles(new(predictions[i].Label.Color, bounding_box_thickness), new[] { predictions[i].Rectangle });
                graphics.DrawString($"{predictions[i].Label.Name} ({score})",
                                new("Consolas", font_size, GraphicsUnit.Pixel), new SolidBrush(predictions[i].Label.Color),
                                new PointF(predictions[i].Rectangle.X, predictions[i].Rectangle.Y));
            }
            return image;
        }
        public static float Area(this RectangleF source)
        {
            return source.Width * source.Height;
        }

        public static Image DrawRotatedBoundingBox(Image image, List<OBBPrediction> predictions, int bounding_box_thickness, int font_size)
        {
            using Graphics graphics = Graphics.FromImage(image);
            for (int i = 0; i < predictions.Count; i++)
            {
                float score = (float)Math.Round(predictions[i].Score, 2);
                graphics.DrawPolygon(new(predictions[i].Label.Color, bounding_box_thickness), GetRotatedPoints(predictions[i]));
                graphics.DrawString($"{predictions[i].Label.Name} ({score})",
                                    new("Consolas", font_size, GraphicsUnit.Pixel), new SolidBrush(predictions[i].Label.Color),
                                    new PointF(predictions[i].Rectangle.X, predictions[i].Rectangle.Y));
            }
            return image;
        }

        public static PointF[] GetRotatedPoints(OBBPrediction prediction)
        {
            PointF[] points = new PointF[]
            {
                new(prediction.Rectangle.X, prediction.Rectangle.Y),
                new(prediction.Rectangle.X + prediction.Rectangle.Width, prediction.Rectangle.Y),
                new(prediction.Rectangle.X + prediction.Rectangle.Width, prediction.Rectangle.Y + prediction.Rectangle.Height),
                new(prediction.Rectangle.X, prediction.Rectangle.Y + prediction.Rectangle.Height)
            };
            OpenCvSharp.Point2f middle = new(prediction.Rectangle.X + prediction.Rectangle.Width * .5f,
                                             prediction.Rectangle.Y + prediction.Rectangle.Height * .5f);
            float cos_angle = (float)Math.Cos(prediction.Angle);
            float sin_angle = (float)Math.Sin(prediction.Angle);
            for (int i = 0; i < 4; i++)
            {
                float offset_x = middle.X - points[i].X;
                float offset_y = middle.Y - points[i].Y;
                float rotated_x = offset_x * cos_angle - offset_y * sin_angle;
                float rotated_y = offset_x * sin_angle + offset_y * cos_angle;
                points[i] = new(middle.X + rotated_x, middle.Y + rotated_y);
            }
            return points;
        }
    }

}