import requests
import zipfile
import os


def download_file_from_google_drive(id, destination):
    URL = url = 'https://drive.google.com/uc?id=177yYDuvWxpt65jn1uWE2YY3ye2_hfA34'
    session = requests.Session()

    response = session.get(URL, params={'id': id}, stream=True)
    token = get_confirm_token(response)

    if token:
        params = {'id': id, 'confirm': token}
        response = session.get(URL, params=params, stream=True)

    save_response_content(response, destination)


def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value

    return None


def save_response_content(response, destination):
    CHUNK_SIZE = 32768

    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk:  # filter out keep-alive new chunks
                f.write(chunk)


def unzip_file(path_to_zip_file, destination):
    with zipfile.ZipFile(path_to_zip_file, 'r') as zip_ref:
        zip_ref.extractall(destination)


def cleanup(path_to_zip_file):
    os.remove(path_to_zip_file)


def main():
    file_id = '14BECMfFtECu6dbaqFXw6cmASVYc2JREM'
    path_to_zip_file = './data/isic18/isic18.zip'
    destination = './data/isic18/'
    download_file_from_google_drive(file_id, path_to_zip_file)
    unzip_file(path_to_zip_file, destination)
    cleanup(path_to_zip_file)


if __name__ == "__main__":
    main()
