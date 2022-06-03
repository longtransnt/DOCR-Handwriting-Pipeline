import './App.css';
import Form from 'react-bootstrap/Form'
import React, { useState } from "react";
import Button from 'react-bootstrap/Button'
import ListGroup from 'react-bootstrap/ListGroup'
import Container from 'react-bootstrap/Container'
import Row from 'react-bootstrap/Row'
import Col from 'react-bootstrap/Col'
import Stack from 'react-bootstrap/Stack'
let imageList = {};

// Import all images in data folder
function importAll(r) {
  imageList = r.keys();
  return r.keys().map(r);
}

const images = importAll(require.context('./../data', false, /\.(png|jpe?g|svg)$/));

// Main application
function App() {
  const [currImage, setCurrImage] = useState(0);
  const [annotation, setAnnotation] = useState('');
  const [annotationList, setAnnotationList] = React.useState([]);
  const [updateState, setUpdateState] = useState(0);
  const [checked, setChecked] = React.useState(false);

  const handleAdd = () => {
    console.log(updateState);
    if (updateState === 1) {
      annotationList[currImage] = imageList[currImage] + "\t" + annotation + "\n";
      setUpdateState(0);
    }
    else {
      // Put function to save new annotation here
      const newList = annotationList.concat(imageList[currImage] + "\t" + annotation + "\n")
      setAnnotationList(newList);
    }

    // Move to next image
    setAnnotation('')
    setCurrImage(currImage + 1);
  }

  const handleListClick = (id) => {
     // Move to this image
     setCurrImage(id);
     setAnnotation("");
     setUpdateState(1);
  }

  function WriteToFile() {
    const element = document.createElement("a");
    console.log(annotationList);
    
    var data = annotationList.filter(function( element ) {
      return element !== undefined;
    });

    const file = new Blob(data, {
      type: "text/plain"
    });
    element.href = URL.createObjectURL(file);
    element.download = "annotation.txt";
    document.body.appendChild(element);
    element.click();
  }

  const handleChange = () => {
    setChecked(!checked);
  };

  return (
    <div className="App">
      <div className="App-header">
      <Button className='upload-btn'>Upload</Button>
      <p>Image Annotation</p>
    
      <Container>
        <Row xs={1} md={2}>
          <Col>
            {/* Displaying Image */}
            <p>Current Image</p>
            <Stack gap={4} className="col-md-11 mx-auto">
              <img id= "currentImage" src={images[currImage]} />
              <p>Annotated Text: {annotationList[currImage] !== undefined ? annotationList[currImage] : "None"}</p>
              <div>
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
                    <div style={{fontSize: '1rem', color: '#005477', fontWeight: '500'}}>
                      <input 
                      style={{marginRight: '5px', color: '#005477', fontWeight: '500'}} 
                      type="checkbox"
                      checked={checked} onChange={handleChange}
                      />
                      Verified by OUCRU
                    </div>
                  </Form.Group>
                </Form>
                <Button variant="primary" onClick={handleAdd}>Save this annotation</Button>{' '}
                <Button variant="primary" onClick={WriteToFile}>Download all annotation</Button>{' '}
              </div>
            </Stack>
          </Col>
          <Col>
            <ListGroup>
              <div id="image-list">
                {imageList.map((im, index) => (
                    <ListGroup.Item 
                    id={"image_" + index} 
                    key={index} 
                    value={index}
                    variant={annotationList[index] !== undefined ? "success" : "danger"} 
                    onClick={(e) => {handleListClick(index)}
                }>
                      <img src= {images[index]} />
                      {im}
                    </ListGroup.Item>
                ))}
              </div>
            </ListGroup>
          </Col>
        </Row>
      </Container>
      </div>
    </div>
  );
}

export default App;

