from Utilize.Data import *
from Utilize.Prediction import *

DEVICE = 'cuda:0'

if __name__ is '__main__':

	dataset = dataset_video(os.path.join(ALL_DATA_SET_FOLDER, 'dataset_video.json'))

	l = len(dataset.video_df)

	mtcnn_model = MTCNN(
    image_size = 100,
    margin=14,
    thresholds = [0.5, 0.6, 0.6],
    keep_all=True,
    device = DEVICE
	)

	dataset.extract_face(10, mtcnn_model, start_index=0, end_index=10)

