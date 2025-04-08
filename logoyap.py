from PIL import Image

def create_icons(input_path, output_base_name="icon"):
    """
    input_path: Orijinal logo dosyasının yolu (örn: 'logo.png').
    output_base_name: Çıktı dosyalarının isim prefix'i (örn: 'icon' -> 'icon-192.png', 'icon-512.png')
    """
    # Oluşturmak istediğin ikon boyutları (PNG)
    sizes = [192, 512]

    # Orijinal resmi Pillow ile aç
    with Image.open(input_path) as img:
        # Transpanlık korumak için RGBA'ya çevir (gerekliyse)
        img = img.convert("RGBA")  

        for size in sizes:
            # İstediğin boyuta göre orantılı biçimde küçült/büyült
            resized_img = img.resize((size, size), resample=Image.Resampling.LANCZOS)


            # Dosya ismini oluştur
            filename = f"{output_base_name}-{size}x{size}.png"
            resized_img.save(filename, format="PNG")
            print(f"'{filename}' oluşturuldu.")

if __name__ == "__main__":
    # Örnek kullanım
    # Elindeki "logo.png" dosyasından "icon-192x192.png" ve "icon-512x512.png" üretir.
    create_icons("logo.png", "icon")
