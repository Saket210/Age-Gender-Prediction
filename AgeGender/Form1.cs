using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Diagnostics;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using System.Windows.Forms;
using OpenCvSharp;
using OpenCvSharp.Dnn;
using OpenCvSharp.Extensions;
using Point = OpenCvSharp.Point;
using Size = OpenCvSharp.Size;

namespace AgeGender
{
    public partial class Form1 : Form
    {
        private bool run = false;
        private bool faceDetect = true;
        private bool ageGender = true;
        private VideoCapture capture;
        private Thread cameraThread;
        private Mat frame;
        private Bitmap image;
        private const int Padding = 10;
        private Net faceNet;
        private Net ageNet;
        private Net genderNet;
        private readonly List<string> _genderList = new List<string> { "Male", "Female" };
        private readonly List<string> ageList = new List<string> { "(0-3)", "(4-7)", "(8-12)", "(15-22)", "(23-30)", "(35-43)", "(45-53)", "(60-100)" };

        public Form1()
        {
            InitializeComponent();
            Load += Form1_Load;
            Closed += Form1_Closed;
        }
        private void Form1_Load(object sender, EventArgs e)
        {
            const string faceProto = "models/deploy.prototxt.txt";
            const string faceModel = "models/res10_300x300_ssd_iter_140000_fp16.caffemodel";
            const string ageModel = "models/age_net.caffemodel";
            const string ageProto = "models/age_deploy.prototxt.txt";
            const string genderModel = "models/gender_net.caffemodel";
            const string genderProto = "models/gender_deploy.prototxt.txt";
            faceNet = CvDnn.ReadNetFromCaffe(faceProto, faceModel);
            ageNet = CvDnn.ReadNetFromCaffe(ageProto, ageModel);
            genderNet = CvDnn.ReadNetFromCaffe(genderProto, genderModel);
        }
        private void Form1_Closed(object sender, EventArgs e)
        {
            cameraThread.Interrupt();
            capture.Release();
        }

        private void CaptureCamera()
        {
            cameraThread = new Thread(new ThreadStart(CaptureCameraCallback));
            cameraThread.Start();
        }

        private void CaptureCameraCallback()
        {
            frame = new Mat();
            capture = new VideoCapture(0);
            capture.Open(0);
            if (capture.IsOpened())
            {
                while (run)
                {
                    capture.Read(frame);
                    if (faceDetect) DetectFaces(frame);
                    image = BitmapConverter.ToBitmap(frame);
                    if (pictureBox.Image != null)
                    {
                        pictureBox.Image.Dispose();
                    }
                    pictureBox.Image = image;
                }
            }
        }

        private void DetectFaces(Mat frame)
        {
            int fh = frame.Rows;
            int fw = frame.Cols;
            var blob = CvDnn.BlobFromImage(frame, 1.0, new Size(300, 300), new Scalar(104, 117, 123), false, false);
            faceNet.SetInput(blob, "data");
            var detection = faceNet.Forward("detection_out");
            var detectionMat = new Mat(detection.Size(2), detection.Size(3), MatType.CV_32F, detection.Ptr(0));

            for (int i = 0; i < detectionMat.Rows; i++)
            {
                float confidence = detectionMat.At<float>(i, 2);
                if (confidence > 0.7)
                {
                    int x1 = (int)(detectionMat.At<float>(i, 3) * fw);
                    int y1 = (int)(detectionMat.At<float>(i, 4) * fh);
                    int x2 = (int)(detectionMat.At<float>(i, 5) * fw);
                    int y2 = (int)(detectionMat.At<float>(i, 6) * fh);

                    Cv2.Rectangle(frame, new Point(x1, y1), new Point(x2, y2), Scalar.Yellow, 2);

                    if (ageGender) AgeGender(x1, y1, x2, y2, fh, fw, frame);
                }
            }
        }

        private void AgeGender(int x1, int y1, int x2, int y2, int fh, int fw, Mat frame)
        {
            //double Paddingh = 0.4 * (y2 - y1);
            //double Paddingw = 0.4 * (x2 - x1);
            var x = x1 - (int)Padding;
            var y = y1 - (int)Padding;
            var w = (x2 - x1) + (int)Padding * 2;
            var h = (y2 - y1) + (int)Padding * 2;
            if (x > 0 && x + w < fw && y > 0 && y + h < fh)

            {
                Rect roiNew = new Rect(x, y, w, h);
                var face = frame[roi: roiNew];


                var meanValues = new Scalar(78.4263377603, 87.7689143744, 114.895847746);
                var blobGender = CvDnn.BlobFromImage(face, 1.0, new Size(227, 227), mean: meanValues,
                    swapRB: false);
                genderNet.SetInput(blobGender);
                var genderPreds = genderNet.Forward();

                GetMaxClass(genderPreds, out int classId, out double classProbGender);
                var gender = _genderList[classId];

                ageNet.SetInput(blobGender);
                var agePreds = ageNet.Forward();
                GetMaxClass(agePreds, out int classIdAge, out double classProbAge);
                var age = ageList[classIdAge];

                var label = $"{gender},{age}";
                Cv2.PutText(frame, label, new Point(x1 - 10, y2 + 20), HersheyFonts.HersheyComplexSmall, 1, Scalar.Yellow, 1);

            }
        }
        private void GetMaxClass(Mat probBlob, out int classId, out double classProb)
        {
            // reshape the blob to 1×1000 matrix
            var probMat = probBlob.Reshape(1, 1);
            Cv2.MinMaxLoc(probMat, out _, out classProb, out _, out var classNumber);
            classId = classNumber.X;
            Debug.WriteLine($"X: {classNumber.X} – Y: {classNumber.Y} ");
        }
        private void button1_Click(object sender, EventArgs e)
        {
            if (button1.Text.Equals("Start"))
            {
                run = true;
                CaptureCamera();
                button1.Text = "Stop";

            }
            else
            {
                run = false;
                //capture.Release();
                button1.Text = "Start";

            }
        }

        
    }
}
