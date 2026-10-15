
import os
import torch
from torch.utils.data import Dataset, DataLoader, distributed
from torchvision import transforms
from torchvision.transforms import functional as TF
import random
import torchvision
import PIL
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

class RandomRotate90or270:
    def __init__(self, p=0.3):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            angle = random.choice([90, 270])
            return TF.rotate(img, angle)
        return img

def build_base_transform(resize: int = 518):
    return [
        transforms.Resize((resize, resize)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ]

def build_train_transform(
    resize=518,
    use_hflip=False,
    use_vflip=False,
    use_rotate90=False,
    use_color_jitter=False,
    use_gray=False,
    use_blur=False,
    # use_random_erasing=False,
):
    ops = []

    if use_hflip:
        ops.append(transforms.RandomHorizontalFlip(p=0.2))

    if use_vflip:
        ops.append(transforms.RandomVerticalFlip(p=0.2))

    if use_rotate90:
        ops.append(RandomRotate90or270(p=0.2))

    if use_color_jitter:
        ops.append(transforms.RandomApply(
                [transforms.ColorJitter(0.3,0.3,0.3,0.05)],
                p=0.2
            )
        )

    if use_gray:
        ops.append(transforms.RandomGrayscale(p=0.1))

    if use_blur:
        ops.append(transforms.RandomApply(
                [transforms.GaussianBlur(kernel_size=23 if resize >= 384 else 11, sigma=(0.1, 2.0))],
                p=0.2
            )
        )

    ops.extend(build_base_transform(resize))

    # if use_random_erasing:
    #     ops.append(transforms.RandomErasing(p=0.25, scale=(0.02,0.15), ratio=(0.3,3.3)))

    return transforms.Compose(ops)

def build_train_transform_new(
    resize=518,
    use_hflip=False,
    use_vflip=False,
    use_rotate90=False,
    use_color_jitter=False,
    use_gray=False,
    use_blur=False,
    p_any=0.3,
):
    candidates = []
    if use_hflip:      
        candidates.append(transforms.RandomHorizontalFlip(p=1.0))
    if use_vflip:      
        candidates.append(transforms.RandomVerticalFlip(p=1.0))
    if use_rotate90:   
        candidates.append(RandomRotate90or270(p=1.0))
    if use_color_jitter:
        candidates.append(transforms.ColorJitter(0.3,0.3,0.3,0.05))
    if use_gray:       
        candidates.append(transforms.Lambda(lambda im: im.convert("L").convert("RGB")))
    if use_blur:
        candidates.append(transforms.GaussianBlur(kernel_size=23 if resize>=384 else 11, sigma=(0.1,2.0)))

    ops = []
    if candidates:
        ops.append(
            transforms.RandomApply(
                [transforms.RandomChoice(candidates)],
                p=p_any
            )
        )

    ops.extend(build_base_transform(resize))
    return transforms.Compose(ops)


def build_train_transform_staged(
    resize=518,
    use_hflip=False,
    use_vflip=False,
    use_rotate90=False,
    use_color_jitter=False,
    use_gray=False,
    use_blur=False,
    p_orient=0.3,
    p_appear=0.3,
):
    ops = []

    orient_candidates = []
    if use_hflip:
        orient_candidates.append(transforms.RandomHorizontalFlip(p=1.0))
    if use_vflip:
        orient_candidates.append(transforms.RandomVerticalFlip(p=1.0))
    if use_rotate90:
        orient_candidates.append(RandomRotate90or270(p=1.0))

    if orient_candidates:
        ops.append(
            transforms.RandomApply(
                [transforms.RandomChoice(orient_candidates)],
                p=p_orient
            )
        )

    appear_candidates = []
    if use_color_jitter:
        appear_candidates.append(transforms.ColorJitter(0.3, 0.3, 0.3, 0.05))  # 确定性
    if use_gray:
        appear_candidates.append(transforms.RandomGrayscale(p=1.0))            # 确定性
    if use_blur:
        ksz = 23 if resize >= 384 else 11
        appear_candidates.append(transforms.GaussianBlur(kernel_size=ksz, sigma=(0.1, 2.0)))

    if appear_candidates:
        ops.append(
            transforms.RandomApply(
                [transforms.RandomChoice(appear_candidates)],
                p=p_appear
            )
        )

    ops.extend(build_base_transform(resize))

    return transforms.Compose(ops)

class TrainDataset(Dataset):
    def __init__(self, root: str, resize=518, **kwargs):
        super().__init__()

        self.root = root
        self.resize = resize
        self.classes = kwargs.get("classnames", ["can"])
        self.samples = []
        self.transform = build_train_transform_staged(
            self.resize,
            use_hflip=kwargs.get("use_hflip", False),
            use_vflip=kwargs.get("use_vflip", False),
            use_rotate90=kwargs.get("use_rotate90", False),
            use_color_jitter=kwargs.get("use_color_jitter", False),
            use_gray=kwargs.get("use_gray", False),
            use_blur=kwargs.get("use_blur", False),
        )

        for classname in self.classes:
            image_dir = os.path.join(
                self.root, classname, "train", "good"
            )

            if not os.path.isdir(image_dir):
                raise FileNotFoundError(
                    f"Training directory not found: {image_dir}"
                )

            image_paths = [
                os.path.join(image_dir, name)
                for name in sorted(os.listdir(image_dir))
                if name.lower().endswith((".png", ".jpg", ".jpeg"))
                and os.path.isfile(os.path.join(image_dir, name))
            ]

            if not image_paths:
                raise RuntimeError(
                    f"No training images found: {image_dir}"
                )

            self.samples.extend(
                (path, classname) for path in image_paths
            )

        print(f"Training image count: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image_path, classname = self.samples[index]

        with PIL.Image.open(image_path) as image:
            image = image.convert("RGB")

        image = self.transform(image)
        return image, classname, image_path
    
class TestDataset(Dataset):

    def __init__(
        self,
        source,
        classname,
        resize=518,
        datasetname = "mvtec",
        **kwargs,
    ):
        super().__init__()
        self.transform_mean = IMAGENET_MEAN
        self.transform_std = IMAGENET_STD
        self.source = source
        self.classnames_to_use = [classname] 
        self.datasetname = datasetname

        self.imgpaths_per_class, self.data_to_iterate = self.get_image_data()

        self.transform_img = transforms.Compose(build_base_transform(resize))

        self.transform_mask = transforms.Compose([
            transforms.Resize(
                (resize, resize),
                interpolation=transforms.InterpolationMode.NEAREST,
            ),
            transforms.ToTensor(),
        ])

        self.imagesize = (3, resize, resize)

    def __getitem__(self, idx):
        classname, anomaly, image_path, mask_path = self.data_to_iterate[idx]
        image = PIL.Image.open(image_path).convert("RGB")
        image = self.transform_img(image)

        if mask_path is not None:
            mask = PIL.Image.open(mask_path).convert("L")
            mask = self.transform_mask(mask)
        else:
            mask = torch.zeros([1, *image.size()[1:]])

        return {
            "image": image,
            "mask": mask,
            "classname": classname,
            "anomaly": anomaly,
            "is_anomaly": int(anomaly not in ("good", "ok")),
            "image_name": "/".join(image_path.split("/")[-4:]),
            "image_path": image_path,
        }

    def __len__(self):
        return len(self.data_to_iterate)

    def get_image_data(self):
        # for new dataset mvtec_ad2
        if self.datasetname == "mvtec_ad2":
            imgpaths_per_class = {}
            data_to_iterate = []

            for classname in self.classnames_to_use:
                test_root = os.path.join(
                    self.source, classname, "test_public"
                )
                imgpaths_per_class[classname] = {}

                for anomaly in ("good", "bad"):
                    image_dir = os.path.join(test_root, anomaly)
                    image_paths = [
                        os.path.join(image_dir, name)
                        for name in sorted(os.listdir(image_dir))
                        if name.lower().endswith((".png", ".jpg", ".jpeg"))
                    ]
                    imgpaths_per_class[classname][anomaly] = image_paths

                    for image_path in image_paths:
                        mask_path = None
                        
                        if anomaly == "bad":
                            stem = os.path.splitext(
                                os.path.basename(image_path)
                            )[0]
                            mask_path = os.path.join(
                                test_root,
                                "ground_truth",
                                "bad",
                                stem + "_mask.png",
                            )
                            if not os.path.isfile(mask_path):
                                raise FileNotFoundError(
                                    f"Unable to find the corresponding exception mask: {mask_path}"
                                )

                        data_to_iterate.append(
                            [classname, anomaly, image_path, mask_path]
                        )
            return imgpaths_per_class, data_to_iterate

        imgpaths_per_class = {}
        maskpaths_per_class = {}

        for classname in self.classnames_to_use:
            classpath = os.path.join(self.source, classname, "test")
            maskpath = os.path.join(self.source, classname, "ground_truth")
            anomaly_types = os.listdir(classpath)

            imgpaths_per_class[classname] = {}
            maskpaths_per_class[classname] = {}

            for anomaly in anomaly_types:
                anomaly_path = os.path.join(classpath, anomaly)
                anomaly_files = sorted(os.listdir(anomaly_path))
                imgpaths_per_class[classname][anomaly] = [
                    os.path.join(anomaly_path, x) for x in anomaly_files
                ]
                if self.datasetname == "mvtec":
                    if  anomaly != "good":
                        anomaly_mask_path = os.path.join(maskpath, anomaly)
                        anomaly_mask_files = sorted(os.listdir(anomaly_mask_path))
                        maskpaths_per_class[classname][anomaly] = [
                            os.path.join(anomaly_mask_path, x) for x in anomaly_mask_files
                        ]
                    else:
                        maskpaths_per_class[classname]["good"] = None
                elif self.datasetname == "visa":
                    if  anomaly != "ok":
                        anomaly_mask_path = os.path.join(maskpath, anomaly)
                        anomaly_mask_files = sorted(os.listdir(anomaly_mask_path))
                        maskpaths_per_class[classname][anomaly] = [
                            os.path.join(anomaly_mask_path, x) for x in anomaly_mask_files
                        ]
                    else:
                        maskpaths_per_class[classname]["ok"] = None

        # Unrolls the data dictionary to an easy-to-iterate list.
        data_to_iterate = []
        for classname in sorted(imgpaths_per_class.keys()):
            for anomaly in sorted(imgpaths_per_class[classname].keys()):
                for i, image_path in enumerate(imgpaths_per_class[classname][anomaly]):
                    data_tuple = [classname, anomaly, image_path]
                    if self.datasetname == "mvtec":
                        if anomaly != "good":
                            data_tuple.append(maskpaths_per_class[classname][anomaly][i])
                        else:
                            data_tuple.append(None)
                    elif self.datasetname == "visa":
                        if anomaly != "ok":
                            data_tuple.append(maskpaths_per_class[classname][anomaly][i])
                        else:
                            data_tuple.append(None)
                    data_to_iterate.append(data_tuple)

        return imgpaths_per_class, data_to_iterate



def build_dataloader(
    mode: str,
    root: str,
    batch_size: int,
    pin_mem: bool = True,
    **kwargs,
):
    """Return (dataset, dataloader, sampler).

    Parameters
    ----------
    mode : str
        "paired" | "mvtec"  —— decides which Dataset subclass to instantiate.
    root : str
        Root path for the dataset. For "paired" this should contain a
        ``train`` subdirectory; for "mvtec" it should be the directory that
        has all class folders (e.g. *mvtec/*).
    **kwargs : dict
        Extra arguments forwarded to the respective dataset constructor.
    """

    if mode == "train":
        dataset = TrainDataset(root=root, **kwargs)
        sampler = distributed.DistributedSampler(
            dataset,
        )
        drop_last = True
    elif mode == "test":
        dataset = TestDataset(source=root, **kwargs)
        sampler = None  # evaluation usually not distributed
        drop_last = False
    else:
        raise ValueError(f"Unsupported mode: {mode}")

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=(sampler is None and mode == "test"),
        pin_memory=pin_mem,
        drop_last=drop_last,
    )

    return dataset, dataloader, sampler
