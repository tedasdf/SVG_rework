import modal

app = modal.App('data_upload')
vol = modal.Volume.from_name("my-volume", create_if_missing=True)


@app.function(volumes={"/data": vol})
def run():
    with open("/data/xyz.txt", "w") as f:
        f.write("hello")
    vol.commit()  # Needed to make sure all changes are persisted before exit