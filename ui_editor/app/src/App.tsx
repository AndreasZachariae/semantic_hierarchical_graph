import { AppBar, Box, CssBaseline, Divider, Icon, IconButton, List, ListItem, ListItemButton, ListItemSecondaryAction, ListItemText, Paper, Stack, Theme, ThemeProvider, Toolbar, Typography, createTheme, makeStyles } from '@mui/material'
import './App.css'
import { AddFloorDialog } from './components/dialogs/AddFloorDialog';
import { Add, Close, Delete, Edit } from '@mui/icons-material';
import { useCallback, useRef, useState } from 'react';
import { useFloors } from './state/EditorContent';
import { useEditorState } from './state/EditorState';
import { EditFloorDialog } from './components/dialogs/EditFloorDialog';
import { AreaRenderer } from './components/renderers/AreaRenderer';
import { LocationRenderer } from './components/renderers/LocationRenderer';

const theme = createTheme({
  palette: {
    mode: 'light',
    primary: {
      main: "#007bff",
    },
    secondary: {
        main: "#93bae2",
    },
  },
});

function App() {
  const [addFloorDialogOpen, setAddFloorDialogOpen] = useState(false);
  const [editFloor, setEditFloor] = useState(-1);

  const { floors, removeFloor } = useFloors();
  const { currentFloor, setCurrentFloor, zoom, setZoom } = useEditorState();

  const currentFloorData = floors.filter(floor => floor.id === currentFloor)[0];

  const pixelsPerMeter = 16;
  const pixelWidth = (currentFloorData?.map?.width || 0) * zoom / (currentFloorData?.map?.resolution || 1) * pixelsPerMeter;
  const pixelHeight = (currentFloorData?.map?.height || 0) * zoom / (currentFloorData?.map?.resolution || 1) * pixelsPerMeter;

  const zoomRef = useRef(zoom);
  zoomRef.current = zoom;

  const [ editAreaWidth, setEditAreaWidth ] = useState(0);
  const [ editAreaHeight, setEditAreaHeight ] = useState(0);

  const onChangeContentElement = useCallback((element: HTMLDivElement | null) => {
    if (!!element) {
      setEditAreaWidth(element.clientWidth);
      setEditAreaHeight(element.clientHeight);
    }

    element?.addEventListener("wheel", (event) => {
      if (event.ctrlKey) {
        event.preventDefault();
        const oldZoom = zoomRef.current;
        const newZoom = oldZoom + Math.sign(event.deltaY) * .01;
        const zoomRatio = newZoom / oldZoom;

        //adjust scroll position for zoom
        const scrollLeft = element.scrollLeft;
        const scrollTop = element.scrollTop;
        const mouseLeft = event.clientX - element.getBoundingClientRect().left;
        const mouseTop = event.clientY - element.getBoundingClientRect().top;

        element.scrollLeft = (scrollLeft + mouseLeft) * zoomRatio - mouseLeft;
        element.scrollTop = (scrollTop + mouseTop) * zoomRatio - mouseTop;
        setZoom(newZoom);
      }
    });
  }, []);

  return (
    <ThemeProvider theme={theme}>
      <div style={{ overflow: "scroll", flexGrow: 1 }} ref={onChangeContentElement}>
        {currentFloor >= 0 && (
          <div style={{position: "relative"}}>
            <img
              src={currentFloorData?.map?.file}
              alt='Floor map'
              style={{
                position: "absolute",
                width: pixelWidth,
                height: pixelHeight,
              }}
            />
            <AreaRenderer
              areas={currentFloorData.areas}
              origin={currentFloorData.map.origin}
              width={currentFloorData.map.width}
              height={currentFloorData.map.height}
              pixelWidth={pixelWidth}
              pixelHeight={pixelHeight}
            />
            <LocationRenderer
              locations={[{ name:"Test", type: 1, position: {x: 50, y: 50, angle: 0} }]}
              origin={currentFloorData.map.origin}
              resolution={currentFloorData.map.resolution}
              width={currentFloorData.map.width}
              height={currentFloorData.map.height}
              pixelWidth={pixelWidth}
              pixelHeight={pixelHeight}
            />
          </div>
        )}
      </div>
      <Paper sx={{ position: "absolute", width: "15rem", height: "20rem", bottom: "2rem", right: "2rem" }}>
        <Stack>
          <Stack direction='row' alignItems='center' gap='.5rem' sx={{ padding: ".5rem", gap: ".5rem" }}>
            <Typography variant='h6'>
              Floors
            </Typography>
            <div style={{ flexGrow: 1 }} />
            <IconButton onClick={() => setAddFloorDialogOpen(true)}>
              <Add/>
            </IconButton>
          </Stack>
          <Divider />
          <List>
            {floors.map(floor => (
              <ListItemButton
                key={floor.id}
                selected={floor.id === currentFloor}
                onClick={() => setCurrentFloor(floor.id || -1)}
              >
                <ListItemText primary={floor.name} />
                <ListItemSecondaryAction>
                  <IconButton onClick={() => setEditFloor(floor.id || -1)}>
                    <Edit/>
                  </IconButton>
                  <IconButton onClick={() => {
                    removeFloor(floor);
                    setCurrentFloor(floors.filter(f => floor.id !== f.id)[0]?.id || -1);
                  }}>
                    <Delete/>
                  </IconButton>
                </ListItemSecondaryAction>
              </ListItemButton>
            ))}
          </List>
        </Stack>
      </Paper>
      <AddFloorDialog open={addFloorDialogOpen} onClose={() => setAddFloorDialogOpen(false)} />
      <EditFloorDialog open={editFloor >= 0} editFloorId={editFloor} onClose={() => setEditFloor(-1)} />
    </ThemeProvider>
  )
}

export default App
