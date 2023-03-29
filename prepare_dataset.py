import glob

class prepare_dataset:
    def __init__(self, image_path, k_fold):
        
        self.image_path = image_path
        self.k_fold = k_fold #fold number for cross-validation

    def train_test_set(self):

        #split good and bad quality images
        good_quality_images = [each for each in glob.glob(self.image_path + '/*') if each.split('/')[-1][:4] == "good"]
        bad_quality_images = [each for each in glob.glob(self.image_path + '/*') if each.split('/')[-1][:4] == "bad_"]

        #create image:label dictionary for dataset load
        good_quality_dict = {each: '0' for each in good_quality_images}
        bad_quality_dict = {each: '1' for each in bad_quality_images}
        all_images_dict = {**good_quality_dict, **bad_quality_dict}

        #learn the number of images for each class
        print("Total good quality image number: ", len(good_quality_images))
        print("Total bad quality image number: ", len(bad_quality_images))

        #split into k folds
        k=self.k_fold
        num_good = len(good_quality_images)
        good_quality_folds = [good_quality_images[x:x+int(num_good/k+1)] for x in range(0, num_good, int(num_good/k + 1))]

        num_bad = len(bad_quality_images)
        bad_quality_folds = [bad_quality_images[x:x+int(num_bad/k+1)] for x in range(0, num_bad, int(num_bad/k + 1))]

        return good_quality_folds, bad_quality_folds