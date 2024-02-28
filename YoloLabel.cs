using System.Drawing;

namespace OpenVinoYOLO
{
    public class YoloLabel(int Id, string Name, Color Color)
    {
        public int Id { get; set; } = Id;

        public string Name { get; set; } = Name;

        public Color Color { get; set; } = Color;
    }
}
