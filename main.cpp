#include <fstream>
#include <sstream>
#include <iostream>
#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

using namespace cv;
using namespace dnn;
using namespace std;

struct Net_config
{
	float person_conf_thres; 
	float person_iou_thres;  
	float kp_conf_thres;  
	float kp_iou_thres;
	float conf_thres_kp_person;
	int overwrite_tol;
	bool use_kp_dets;
};

int endswith(string s, string sub) {
	return s.rfind(sub) == (s.length() - sub.length()) ? 1 : 0;
}

int kp_face[5] = { 0, 1, 2, 3, 4 };
int segments[12][2] = { {5, 6}, {5, 11}, {11, 12}, {12, 6}, {5, 7}, {7, 9}, {6, 8}, {8, 10}, {11, 13}, {13, 15}, {12, 14}, {14, 16} };
int crowd_segments[14][2] = { {0, 13}, {1, 13}, {0, 2}, {2, 4}, {1, 3}, {3, 5}, {0, 6}, {6, 7}, {7, 1}, {6, 8}, {8, 10}, {7, 9}, {9, 11}, {12, 13} };

class YOLO
{
public:
	YOLO(Net_config config, string modelpath);
	void detect(Mat& frame);
private:
	const int inpWidth = 1280;
	const int inpHeight = 1280;
	const int num_stride = 4;
	vector<string> class_names;
	int num_class;
	int num_lines;
	int num_face_pts;
	int* plines;

	Net_config config;
	const bool keep_ratio = true;
	Net net;
	void drawPred(float conf, int left, int top, int right, int bottom, Mat& frame, int classid);
	Mat resize_image(Mat srcimg, int *newh, int *neww, int *top, int *left);
	const float anchors[4][6] = { {19, 27, 44, 40, 38, 94},{96, 68, 86, 152, 180, 137},{140, 301, 303, 264, 238, 542},
					   {436, 615, 739, 380, 925, 792} };
};

YOLO::YOLO(Net_config config, string modelpath)
{
	this->config.person_conf_thres = config.person_conf_thres;
	this->config.person_iou_thres = config.person_iou_thres;
	this->config.kp_conf_thres = config.kp_conf_thres;
	this->config.kp_iou_thres = config.kp_iou_thres;
	this->config.conf_thres_kp_person = config.conf_thres_kp_person;
	this->config.overwrite_tol = config.overwrite_tol;
	this->config.use_kp_dets = config.use_kp_dets;

	this->net = readNet(modelpath);
	if (endswith(modelpath, "_coco.onnx"))
	{
		ifstream ifs("class.names");
		string line;
		while (getline(ifs, line)) this->class_names.push_back(line);
		this->num_lines = 12;
		this->num_face_pts = 5;
		plines = (int*)segments;
	}
	else
	{
		ifstream ifs("crowd_class.names");
		string line;
		while (getline(ifs, line)) this->class_names.push_back(line);
		this->num_lines = 14;
		this->num_face_pts = 0;
		plines = (int*)crowd_segments;
	}
	this->num_class = class_names.size();
}

Mat YOLO::resize_image(Mat srcimg, int *newh, int *neww, int *top, int *left)
{
	int srch = srcimg.rows, srcw = srcimg.cols;
	*newh = this->inpHeight;
	*neww = this->inpWidth;
	Mat dstimg;
	if (this->keep_ratio && srch != srcw) {
		float hw_scale = (float)srch / srcw;
		if (hw_scale > 1) {
			*newh = this->inpHeight;
			*neww = int(this->inpWidth / hw_scale);
			resize(srcimg, dstimg, Size(*neww, *newh), INTER_AREA);
			*left = int((this->inpWidth - *neww) * 0.5);
			copyMakeBorder(dstimg, dstimg, 0, 0, *left, this->inpWidth - *neww - *left, BORDER_CONSTANT, 114);
		}
		else {
			*newh = (int)this->inpHeight * hw_scale;
			*neww = this->inpWidth;
			resize(srcimg, dstimg, Size(*neww, *newh), INTER_AREA);
			*top = (int)(this->inpHeight - *newh) * 0.5;
			copyMakeBorder(dstimg, dstimg, *top, this->inpHeight - *newh - *top, 0, 0, BORDER_CONSTANT, 114);
		}
	}
	else {
		resize(srcimg, dstimg, Size(*neww, *newh), INTER_AREA);
	}
	return dstimg;
}

void YOLO::drawPred(float conf, int left, int top, int right, int bottom, Mat& frame, int classid)   // Draw the predicted bounding box
{
	//Draw a rectangle displaying the bounding box
	rectangle(frame, Point(left, top), Point(right, bottom), Scalar(0, 0, 255), 1);

	//Get the label for the class name and its confidence
	string label = format("%.2f", conf);
	label = this->class_names[classid] + ":" + label;

	//Display the label at the top of the bounding box
	int baseLine;
	Size labelSize = getTextSize(label, FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
	top = max(top, labelSize.height);
	//rectangle(frame, Point(left, top - int(1.5 * labelSize.height)), Point(left + int(1.5 * labelSize.width), top + baseLine), Scalar(0, 255, 0), FILLED);
	putText(frame, label, Point(left, top), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(0, 255, 0), 1);
}

void YOLO::detect(Mat& frame)
{
	int newh = 0, neww = 0, padh = 0, padw = 0;
	Mat dstimg = this->resize_image(frame, &newh, &neww, &padh, &padw);
	Mat blob = blobFromImage(dstimg, 1 / 255.0, Size(this->inpWidth, this->inpHeight), Scalar(0, 0, 0), true, false);
	this->net.setInput(blob);
	vector<Mat> outs;
	this->net.forward(outs, this->net.getUnconnectedOutLayersNames());

	int num_proposal = outs[0].size[1];
	int nout = outs[0].size[2];
	if (outs[0].dims > 2)
	{
		outs[0] = outs[0].reshape(0, num_proposal);
	}
	const int num_coords = (nout - this->num_class - 5) * 0.5;
	/////generate proposals
	vector<float> person_confidences;
	vector<Rect> person_boxes;
	vector<int> person_classIds;
	vector<float> kp_confidences;
	vector<Rect> kp_boxes;
	vector<int> kp_classIds;
	vector<vector<float>> poses;

	float ratioh = (float)frame.rows / newh, ratiow = (float)frame.cols / neww;
	int n = 0, q = 0, i = 0, j = 0, k = 0, row_ind = 0; ///xmin,ymin,xamx,ymax, box_score, class_score, num_coords*2
	float* pdata = (float*)outs[0].data;
	for (n = 0; n < this->num_stride; n++)   ///ÌØÕ÷Í¼³ß¶È
	{
		const float stride = pow(2, n + 3);
		int num_grid_x = (int)ceil((this->inpWidth / stride));
		int num_grid_y = (int)ceil((this->inpHeight / stride));
		for (q = 0; q < 3; q++)    ///anchor
		{
			const float anchor_w = this->anchors[n][q * 2];
			const float anchor_h = this->anchors[n][q * 2 + 1];
			for (i = 0; i < num_grid_y; i++)
			{
				for (j = 0; j < num_grid_x; j++)
				{
					float box_score = pdata[4];
					Mat scores = outs[0].row(row_ind).colRange(5, 5 + this->num_class);
					Point classIdPoint;
					double max_class_socre;
					// Get the value and location of the maximum score
					minMaxLoc(scores, 0, &max_class_socre, 0, &classIdPoint);
					max_class_socre *= box_score;
					const int class_idx = classIdPoint.x;

					if (class_idx == 0)
					{
						if (box_score > this->config.person_conf_thres && max_class_socre > this->config.person_conf_thres)
						{
							float cx = (pdata[0] * 2.f - 0.5f + j) * stride;  ///cx
							float cy = (pdata[1] * 2.f - 0.5f + i) * stride;   ///cy
							float w = powf(pdata[2] * 2.f, 2.f) * anchor_w;   ///w
							float h = powf(pdata[3] * 2.f, 2.f) * anchor_h;  ///h

							int left = int((cx - padw - 0.5 * w)*ratiow);
							int top = int((cy - padh - 0.5 * h)*ratioh);

							person_confidences.push_back((float)max_class_socre);
							person_boxes.push_back(Rect(left, top, (int)(w*ratiow), (int)(h*ratioh)));
							person_classIds.push_back(class_idx);
							
							vector<float> kp(num_coords * 3, 0);
							for (k = 0; k < num_coords; k++)
							{
								float x = pdata[5 + this->num_class + 2 * k] * 4 - 2;
								float y = pdata[5 + this->num_class + 2 * k + 1] * 4 - 2;
								x *= anchor_w;
								y *= anchor_h;
								x += j * stride;
								y += i * stride;
								x = (x - padw)*ratiow;
								y = (y - padh)*ratioh;
								kp[k * 3] = x;
								kp[k * 3 + 1] = y;
							}
							poses.push_back(kp);
						}
					}
					else
					{
						if (box_score > this->config.kp_conf_thres && max_class_socre > this->config.kp_conf_thres)
						{
							float cx = (pdata[0] * 2.f - 0.5f + j) * stride;  ///cx
							float cy = (pdata[1] * 2.f - 0.5f + i) * stride;   ///cy
							float w = powf(pdata[2] * 2.f, 2.f) * anchor_w;   ///w
							float h = powf(pdata[3] * 2.f, 2.f) * anchor_h;  ///h

							int left = int((cx - padw - 0.5 * w)*ratiow);
							int top = int((cy - padh - 0.5 * h)*ratioh);

							kp_confidences.push_back((float)max_class_socre);
							kp_boxes.push_back(Rect(left, top, (int)(w*ratiow), (int)(h*ratioh)));
							kp_classIds.push_back(class_idx);
						}				
					}
					row_ind++;
					pdata += nout;
				}
			}
		}
	}

	// Perform non maximum suppression to eliminate redundant overlapping boxes with
	// lower confidences
	vector<int> person_indices;
	dnn::NMSBoxes(person_boxes, person_confidences, this->config.person_conf_thres, this->config.person_iou_thres, person_indices);
	vector<int> kp_indices;
	dnn::NMSBoxes(kp_boxes, kp_confidences, this->config.kp_conf_thres, this->config.kp_iou_thres, kp_indices);
	vector<int> pose_mask;
	for (i = 0; i < person_indices.size(); i++)
	{
		const int person_id = person_indices[i];
		if (person_confidences[person_id] > this->config.conf_thres_kp_person)
		{
			pose_mask.push_back(person_id);
		}
	}

	for (i = 0; i < kp_indices.size(); i++)
	{
		int idx = kp_indices[i];
		Rect box = kp_boxes[idx];
		float x = box.x + box.width * 0.5;
		float y = box.y + box.height * 0.5;
		float conf = kp_confidences[idx];
		int pt_id = kp_classIds[idx] - 1;
		int min_id = 0;
		float min_dist = 10000;
		for (j = 0; j < pose_mask.size(); j++)
		{
			const int pose_id = pose_mask[j];
			const float dist = sqrt(powf(poses[pose_id][pt_id * 3] - x, 2) + powf(poses[pose_id][pt_id * 3 + 1] - y, 2));
			if (dist < min_dist)
			{
				min_dist = dist;
				min_id = pose_id;
			}
		}
		if (conf > poses[min_id][pt_id * 3 + 2] && min_dist < this->config.overwrite_tol)
		{
			poses[min_id][pt_id * 3] = x;
			poses[min_id][pt_id * 3 + 1] = y;
			poses[min_id][pt_id * 3 + 2] = conf;
		}
	}

	for (i = 0; i < person_indices.size(); ++i)
	{
		int idx = person_indices[i];
		Rect box = person_boxes[idx];
		this->drawPred(person_confidences[idx], box.x, box.y,
			box.x + box.width, box.y + box.height, frame, person_classIds[idx]);
	}
	for (i = 0; i < pose_mask.size(); i++)
	{
		for (j = 0; j < num_coords; j++)
		{
			if (poses[pose_mask[i]][j * 3 + 2] > 0)
			{
				circle(frame, Point(int(poses[pose_mask[i]][j * 3]), int(poses[pose_mask[i]][j * 3 + 1])), 1, Scalar(0, 255, 0), -1);
			}
		}
		for (j = 0; j < this->num_lines; j++)
		{
			Point pt1 = Point(int(poses[pose_mask[i]][this->plines[2 * j] * 3]), int(poses[pose_mask[i]][this->plines[2 * j] * 3 + 1]));
			Point pt2 = Point(int(poses[pose_mask[i]][this->plines[2 * j + 1] * 3]), int(poses[pose_mask[i]][this->plines[2 * j + 1] * 3 + 1]));
			line(frame, pt1, pt2, Scalar(255, 0, 255), 1);
		}
		/*for (j = 0; j < this->num_face_pts; j++)
		{
			circle(frame, Point(int(poses[pose_mask[i]][kp_face[j] * 3]), int(poses[pose_mask[i]][kp_face[j] * 3 + 1])), 1, Scalar(255, 0, 255), -1);
		}*/
	}
}

int main()
{
	Net_config yolo_nets = { 0.7, 0.45, 0.5, 0.45, 0.2, 25,true };
	YOLO yolo_model(yolo_nets, "weights/kapao_m_coco.onnx");
	string imgpath = "images/crowdpose_100024.jpg";
	Mat srcimg = imread(imgpath);
	yolo_model.detect(srcimg);

	static const string kWinName = "Deep learning object detection in OpenCV";
	namedWindow(kWinName, WINDOW_NORMAL);
	imshow(kWinName, srcimg);
	waitKey(0);
	destroyAllWindows();
}