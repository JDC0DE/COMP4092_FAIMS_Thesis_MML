import React, { useState, useEffect, Fragment } from "react";
import * as tf from "@tensorflow/tfjs";
import { DropzoneArea } from "mui-file-dropzone";
import { Backdrop, Chip, CircularProgress, Grid, Stack } from "@mui/material";
import { math } from "@tensorflow/tfjs";

function App() {
  const [model, setModel] = useState(null);
  const [classLabels, setClassLabels] = useState(null);
  const [loading, setLoading] = useState(false);
  const [confidence, setConfidence] = useState(null);
  const [predictedClass, setPredictedClass] = useState(null);

  useEffect(() => {
    const loadModel = async () => {
      //const model_url = "{hosting_url}tfjs/MobileNetV3Large/model.json";
      //C:/Users/joshu/OneDrive/Documents/Uni/COMP4092/COMP4092_FAIMS_Thesis_MML/Notes/imgapp/tfjs/MobileNetV3Large/model.json"
      //const model = await tf.loadGraphModel("InceptionV3/model.json");
      const model = await tf.loadGraphModel("MobileNetV3Large/model.json");
      //const model = await tf.loadLayersModel("Sequential_layers/model.json");
      
      //model.predict().;
      
      console.log(model.modelVersion);

      setModel(model);
    };

    const getClassLabels = async () => {
      const res = await fetch(
        "https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json"
      );

      const data = await res.json();
      

      setClassLabels(data);
    };

    loadModel();
    getClassLabels();
  }, []);

  const readImageFile = (file) => {
    return new Promise((resolve) => {
      const reader = new FileReader();

      reader.onload = () => resolve(reader.result);

      reader.readAsDataURL(file);
    });
  };

  const createHTMLImageElement = (imageSrc) => {
    return new Promise((resolve) => {
      const img = new Image();

      img.onload = () => resolve(img);

      img.src = imageSrc;
    });
  };

  const handleImageChange = async (files) => {
    if (files.length === 0) {
      setConfidence(null);
      setPredictedClass(null);
    }

    if (files.length === 1) {
      setLoading(true);
      
      
      const imageSrc = await readImageFile(files[0]);
      const image = await createHTMLImageElement(imageSrc);

      // tf.tidy for automatic memory cleanup
      const [predictedClass, confidence] = tf.tidy(() => {
        const tensorImg = tf.browser.fromPixels(image).resizeNearestNeighbor([224, 224]).toFloat().expandDims();
        const result = model.predict(tensorImg);

        const predictions = result.dataSync();
        const predicted_index = result.as1D().argMax().dataSync()[0];
        //const predicted_index = result.array()[0][0];

        const predictedClass = classLabels[predicted_index];
        const confidence = Math.round(predictions[predicted_index] * 100);
        
        //const confidence = 100 * Math.max(tf.softmax(predicted_index));

        return [predictedClass, confidence];
      });

      setPredictedClass(predictedClass);
      setConfidence(confidence);
      setLoading(false);
    }
  };

  return (
    <Fragment>
      <Grid container className="App" direction="column" alignItems="center" justifyContent="center" marginTop="12%">
        <Grid item>
          <h1 style={{ textAlign: "center", marginBottom: "1.5em" }}>MobileNetV3 Demo</h1>
          <DropzoneArea
            acceptedFiles={["image/*"]}
            dropzoneText={"Add an image"}
            onChange={handleImageChange}
            maxFileSize={10000000}
            filesLimit={1}
            showAlerts={["error"]}
          />
          <Stack style={{ marginTop: "2em", width: "12rem" }} direction="row" spacing={1}>
            <Chip
              label={predictedClass === null ? "Prediction:" : `Prediction: ${predictedClass}`}
              style={{ justifyContent: "left" }}
              variant="outlined"
            />
            <Chip
              label={confidence === null ? "Confidence:" : `Confidence: ${confidence}%`}
              style={{ justifyContent: "left" }}
              variant="outlined"
            />
          </Stack>
        </Grid>
      </Grid>

      <Backdrop sx={{ color: "#fff", zIndex: (theme) => theme.zIndex.drawer + 1 }} open={loading}>
        <CircularProgress color="inherit" />
      </Backdrop>
    </Fragment>
  );
}

export default App;