if __name__ == "__main__":
    from prepare_mini_coco import ImageTokenizer

    img_enc = ImageTokenizer()

    tokens = img_enc.encode("project/chameleon/test_image.jpeg")

    for token in tokens:
        print(token)