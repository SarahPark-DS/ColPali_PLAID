import matplotlib as plt

#%% 이미지 샘플 확인
def show_image(idx, ds=ds):
    """인덱스로 이미지와 관련 정보 시각화"""
    
    sample = ds[idx]
    img = sample['image'].convert("RGB")
    
    fig, ax = plt.subplots(figsize=(8, 10))
    ax.imshow(img)
    ax.axis('off')
    ax.set_title(
        f"[{idx}] {sample['image_filename']}\n{sample['query']}",
        fontsize=10, wrap=True
    )
    plt.tight_layout()
    plt.show()
