import './App.css';
import 'react-dropdown/style.css';
import 'react-toastify/dist/ReactToastify.css';

import Form from 'react-bootstrap/Form'
import React, { useState, useCallback, useEffect } from "react";
import ListGroup from 'react-bootstrap/ListGroup'
import Container from 'react-bootstrap/Container'
import Row from 'react-bootstrap/Row'
import Col from 'react-bootstrap/Col'
import Stack from 'react-bootstrap/Stack'
import ImageUpload from './components/ImageUpload';
import { ToastContainer, toast } from 'react-toastify';
import { Scrollbars } from 'react-custom-scrollbars'
import { SplitButton, Dropdown } from 'react-bootstrap'
import ImageGallery from 'react-image-gallery';

let imageList = {};
let verified = {};

// Import all images in data folder
function importAll(r) {
  imageList = r.keys();
  return r.keys().map(r);
}

function transformUploads(uploads) {
  return uploads.map(u => ({
    original: u.imageUrl,
    thumbnail: u.thumbnailUrl
  }));
}

const images = importAll(require.context('./../data', false, /\.(png|jpe?g|svg)$/));

const notiSaving = () => toast.warn('Please input annotation before saving!', {
  position: "top-right",
  autoClose: 2000,
  hideProgressBar: false,
  closeOnClick: true,
  pauseOnHover: true,
  draggable: true,
  progress: undefined,
});

const notiDownload = () => toast.warn('Required at least 1 annotation to download!', {
  position: "top-right",
  autoClose: 2000,
  hideProgressBar: false,
  closeOnClick: true,
  pauseOnHover: true,
  draggable: true,
  progress: undefined,
});
// Main application
function App() {
  const [currImage, setCurrImage] = useState(0);
  const [annotation, setAnnotation] = useState('');
  const [annotationList, setAnnotationList] = useState([]);
  const [updateState, setUpdateState] = useState(0);
  const [checked, setChecked] = useState(false);
  const [image, setImage] = useState({});

  const fetchUploads = useCallback(() => {
    fetch('http://annotationnode-env.eba-iv5i9cmp.us-west-2.elasticbeanstalk.com/api/uploads')
      .then(response => response.json().then(data => setImage(transformUploads(data))))
      .catch(console.error)
  }, []);

  useEffect(() => {
    fetchUploads();
  }, [fetchUploads])

  const handleAdd = () => {
    console.log(updateState);
    if (annotation !==  '') {
      if (updateState === 1 ) {
        annotationList[currImage] = image[currImage] + "\t" + annotation + "\n";
        verified[currImage] = checked;
        setUpdateState(0);
      }
      else {
        // Put function to save new annotation here
        const newList = annotationList.concat(image[currImage] + "\t" + annotation + "\n")
        setAnnotationList(newList);
      }
    // Move to next image
    setAnnotation('')
    setCurrImage(currImage + 1);
    } else {
      setUpdateState(0);
      notiSaving(); // not allow to save if no annotation
    }
  }

  const handleListClick = (id) => {
     // Move to this image
     setCurrImage(id);
     setAnnotation("");
     setUpdateState(1);
     setChecked(false);
  }

  function WriteToFile() {
    const element = document.createElement("a");
    console.log(annotationList);
    if (annotationList.length === 0) {
      notiDownload()
    } else {
      var data = annotationList.filter(function( element ) {
        return (element !== undefined && verified[annotationList.indexOf(element)]); // only verified annotation can be downloaded
      });
      const file = new Blob(data, {
        type: "text/plain"
      });
      element.href = URL.createObjectURL(file);
      element.download = "annotation.txt";
      document.body.appendChild(element);
      element.click();
    }
  }

  const handleChecked = () => {
    setChecked(!checked)
  };

  return (
    <div className="App">
      <div className="App-header">
        <button className='upload-btn'>Upload</button>
      <Container>
        <Row xs={1} md={2}>
          <Col>
            {/* Displaying Image */}
            <p style={{textAlign: 'center'}}>Current Image</p>
            <Stack gap={4} className="col-md-11 mx-auto">
              <img id= "currentImage" src={image[currImage]} />
              <p style={{fontSize: '22px'}}>Annotation Preview: 
                <span style={{fontSize: '18px', paddingLeft: '5px'}}>
                  {/* Preview annotation */}
                  {annotationList[currImage] !== undefined ? annotationList[currImage].split(image[currImage] + "\t") : annotation !== '' ? annotation : "None"}
                </span>
              </p>
              <Form onSubmit={handleAdd}>
                <Form.Group className="mb-3" controlId="formBasicEmail">
                  <Form.Control 
                  as="textarea" 
                  rows={4} 
                  placeholder="Enter your annotation here." 
                  style={{border: 'none', height: '20vh'}}
                  value={annotation} 
                  onChange={(e) => {setAnnotation(e.target.value)}}
                  />
                  <div style={{fontSize: '1rem', color: '#005477', fontWeight: '500', float: 'right'}}>
                    <input 
                    id='checkbox3'
                    style={{margin: '5px 5px 0 0', color: '#005477', fontWeight: '500', cursor: 'pointer'}} 
                    type="checkbox"
                    checked={checked} onChange={handleChecked}
                    />
                    <label htmlFor="verified" onClick={handleChecked} style={{cursor: 'pointer'}}>Verified by OUCRU</label>
                  </div>
                  <div style={{fontSize: '1rem', color: '#005477', float: 'left'}}>
                    <label>% Confidence</label>
                    <input type="text"
                      style={{margin: '5px 0 0 5px', color: '#005477', width: '3vw', border: 'none'}}
                    />
                  </div>
                </Form.Group>
              </Form>
            </Stack>
          </Col>
          <Col>
            {/* Image List */}
            <p style={{textAlign: 'center'}}>Image List</p>
            <Scrollbars>
              <div id="image-list">
                {imageList.map((im, index) => (
                    <ListGroup.Item 
                      id={"image_" + index} 
                      key={index} 
                      value={index}
                      variant={
                        annotationList[index] === undefined ? "danger" : 
                        verified[index] === false ? "warning" : "success"
                      } 
                      style={{cursor: 'pointer'}}
                      onClick={(e) => {handleListClick(index)}}
                    >
                      {image && image.length ? (
                        <img src= {JSON.stringify(image)} />
                      ) : null}
                      {im}
                    </ListGroup.Item>
                ))}
              </div>
            </Scrollbars>
          </Col>
        </Row>
      </Container>
      <Row style={{marginTop: '7rem'}}>
          <Col>
            <div style={{float: 'left'}}>
              <ImageUpload />
            </div>
            <div style={{float: 'right'}}>
              <button className='save-btn' onClick={handleAdd}>Save the annotation</button>{' '}
              <div className='download-btn'>
              <SplitButton 
                id={`split-button-basic`}
                title={'Download'}
                
              >
                <Dropdown.Item eventKey="1">0% Confidence</Dropdown.Item>
                <Dropdown.Item eventKey="2">25% Confidence</Dropdown.Item>
                <Dropdown.Item eventKey="3">50% Confidence</Dropdown.Item>
                <Dropdown.Item eventKey="4">75% Confidence</Dropdown.Item>
                <Dropdown.Item eventKey="5">100% Confidence</Dropdown.Item>
                <Dropdown.Divider />
                <Dropdown.Item eventKey="6" onClick={WriteToFile}>Download All</Dropdown.Item>
              </SplitButton>
              </div>
            </div>
          </Col>
        </Row>
      </div>
      <ToastContainer style={{width: '20vw'}} />
    </div>
  );
}

export default App;

