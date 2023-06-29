
import { Upload } from '@mui/icons-material';
import { Box, Button, DialogActions, DialogContent, DialogContentText, DialogTitle, FormControl, FormControlLabel, FormLabel, IconButton, Input, InputAdornment, InputLabel, Paper, Stack, TextField } from '@mui/material';
import Dialog from '@mui/material/Dialog';
import { useEffect, useState } from 'react';
import { useFloors } from '../../state/EditorContent';
import { useEditorState } from '../../state/EditorState';

/**
 * Reads a file to a base64 string
 * @param file 
 * @returns 
 */
function readPngToBase64(file: File): Promise<string> {
    return new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.readAsDataURL(file);
        reader.onload = e => resolve(e.target?.result as string);
        reader.onerror = error => reject(error);
    });
}

/**
 * Reads the file to a JSON object
 * @param file
 * @returns 
 */
function readJson(file: File): Promise<any> {
    return new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.readAsText(file);
        reader.onload = e => resolve(JSON.parse(e.target?.result as string));
        reader.onerror = error => reject(error);
    });
}

interface AddFloorDialogProps {
    open: boolean;
    onClose: () => void;
};

export function AddFloorDialog(props: AddFloorDialogProps) {

    const { addFloor } = useFloors();
    const { setCurrentFloor } = useEditorState();

    const [mapFile, setMapFile] = useState<File | null>(null);
    const [metadataFile, setMetadataFile] = useState<File | null>(null);
    const [name, setName] = useState<string>('');

    useEffect(() => {
        if (!props.open) {
            setMapFile(null);
            setMetadataFile(null);
            setName('');
        }
    }, [ props.open ]);

    return (
        <Dialog open={props.open} onClose={props.onClose} fullWidth>
            <DialogTitle>Add Floor</DialogTitle>
            <DialogContent>
                <Box sx={{ display: 'flex', flexDirection: 'column', justifyContent: 'space-between', paddingTop: "1rem", gap: "1rem" }}>
                    <TextField
                        label="Name"
                        variant="outlined"
                        value={name}
                        onChange={(e) => setName(e.target.value)}
                    />
                    <FormControl fullWidth>
                        <FormLabel>Select Map</FormLabel>
                        <Paper variant='outlined'>
                            <Stack direction='row' alignItems='center' gap="1rem" sx={{ padding: '1rem' }}>
                                <Upload color='primary' fontSize='medium'/>
                                <input type='file' accept='.png,.jpg,.jpeg,.svg' onChange={(e) => { setMapFile(e.target.files?.item(0) ?? null) }} />
                            </Stack>
                        </Paper>
                    </FormControl>
                    <FormControl fullWidth>
                        <FormLabel>Select Metadata</FormLabel>
                        <Paper variant='outlined'>
                            <Stack direction='row' alignItems='center' gap="1rem" sx={{ padding: '1rem' }}>
                                <Upload color='primary' fontSize='medium'/>
                                <input type='file' accept='.json' onChange={(e) => { setMetadataFile(e.target.files?.item(0) ?? null) }} />
                            </Stack>
                        </Paper>
                    </FormControl>
                </Box>
            </DialogContent>
            <DialogActions>
                <Button onClick={props.onClose} color="inherit" >Cancel</Button>
                <Button
                    onClick={async () => {
                        const mapFileContent = await readPngToBase64(mapFile as File);
                        const mapMetadata = await readJson(metadataFile as File);
                        addFloor({
                            name: name,
                            map: {
                                file: mapFileContent,
                                origin: {
                                    x: mapMetadata.origin.x,
                                    y: mapMetadata.origin.y,
                                    angle: mapMetadata.origin.angle,
                                },
                                resolution: mapMetadata.resolution,
                                width: mapMetadata.width,
                                height: mapMetadata.height,
                            },
                            areas: [],
                            locations: []
                        }).then(f => setCurrentFloor(f.id || -1)).then(props.onClose);
                    }}
                    color="primary"
                    disabled={mapFile === null || metadataFile === null || name === ''}
                >Add</Button>
            </DialogActions>
        </Dialog>
    );
}