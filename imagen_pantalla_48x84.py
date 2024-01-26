import sys
import struct

def bitmap_to_hex_array(filename):
    with open(filename, 'rb') as f:
        # Read the BMP header
        header = f.read(54)

        # Check if it's a valid BMP image
        if header[:2] != b'BM':
            print('Invalid BMP image')
            sys.exit(1)

        # Get the image width and height
        width, height = struct.unpack('<II', header[18:26])

        # Check if the image dimensions are correct
        if width != 84 or height != 48:
            print('Invalid image dimensions')
            sys.exit(1)

        # Read the image data
        image_data = f.read(width * height * 3)

        # Convert the image data to hex codes
        hex_array = []
        for i in range(height):
            row_data = image_data[i * width * 3:(i + 1) * width * 3]
            row_hex = []
            for j in range(0, len(row_data), 3):
                pixel_data = row_data[j:j + 3]
                pixel_hex = struct.pack('<BBB', *pixel_data)
                row_hex.append(pixel_hex)

            hex_array.append(row_hex)

        return hex_array

def main():
    if len(sys.argv) != 2:
        print('Usage: imagen_micro.py <filename>')
        sys.exit(1)

    filename = sys.argv[1]
    hex_array = bitmap_to_hex_array(filename)

    for row in hex_array:
        for byte in row:
            print(f'0x{byte.hex()}, ', end='')
        print()

if __name__ == '__main__':
    main()
