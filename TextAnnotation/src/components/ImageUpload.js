import React, { useState, useEffect } from 'react';
import Lightbox from 'react-image-lightbox';
import 'react-image-lightbox/style.css';

function ImageUpload() {
  const [isOpen, setIsOpen] = useState(false);
  const [selectedImage, setSelectedImage] = useState()
  const [preview, setPreview] = useState()

    // create a preview as a side effect, whenever selected file is changed
    useEffect(() => {
        if (!selectedImage) {
            setPreview(undefined)
            return
        }
        const objectUrl = URL.createObjectURL(selectedImage)
        setPreview(objectUrl)
        // free memory when ever this component is unmounted
        return () => URL.revokeObjectURL(objectUrl)
    }, [selectedImage])

    const onSelectFile = e => {
        if (!e.target.files || e.target.files.length === 0) {
            setSelectedImage(undefined)
            return
        }
        setSelectedImage(e.target.files[0])
    }

  return (
    <div>
        {/* <input type='file' onChange={onSelectFile} /> */}
        <button className='view-image-btn' onClick={() => setIsOpen(true)}>View the original image</button>
        {isOpen && (
        <Lightbox
            mainSrc={preview}
            onCloseRequest={() => setIsOpen(false)}
        />
        )}
    </div>
  );
}

export default ImageUpload
