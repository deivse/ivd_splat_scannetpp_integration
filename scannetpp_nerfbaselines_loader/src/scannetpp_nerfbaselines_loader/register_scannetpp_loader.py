from nerfbaselines import register

register(
    {
        "id": "scannet++",
        "load_dataset_function": "scannetpp_nerfbaselines_loader.scannetpp_loader:scannetpp_loader_regular",
    }
)

register(
    {
        "id": "eval_on_train_set_scannet++",
        "load_dataset_function": "scannetpp_nerfbaselines_loader.scannetpp_loader:scannetpp_loader_test_from_train_set",
    }
)

register(
    {
        "id": "scannet++",
        "download_dataset_function": "scannetpp_nerfbaselines_loader.scannetpp_loader:download_scannetpp_not_implemented",
        "evaluation_protocol": "default",
        "metadata": {},
    }
)

register(
    {
        "id": "eval_on_train_set_scannet++",
        "download_dataset_function": "scannetpp_nerfbaselines_loader.scannetpp_loader:download_scannetpp_not_implemented",
        "evaluation_protocol": "default",
        "metadata": {},
    }
)
