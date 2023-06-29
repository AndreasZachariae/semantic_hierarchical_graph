import { Box, Button, Dialog, DialogActions, DialogContent, DialogTitle, TextField } from "@mui/material";
import { useFloors } from "../../state/EditorContent";
import { useEffect, useState } from "react";

interface EditFloorDialogProps {
    open: boolean;
    editFloorId: number;
    onClose: () => void;
};

export function EditFloorDialog(props: EditFloorDialogProps) {
    const { floors, setFloor } = useFloors();
    const currentFloorData = floors.find(floor => floor.id === props.editFloorId);

    const [name, setName] = useState('');
    useEffect(() => {
        if (props.open) {
            setName(currentFloorData?.name || '');
        }
    }, [ props.open ])

    return (
        <Dialog
            open={props.open}
            onClose={props.onClose}
        >
            <DialogTitle>Edit Floor</DialogTitle>
            <DialogContent>
                <Box sx={{ display: 'flex', flexDirection: 'column', justifyContent: 'space-between', paddingTop: "1rem", gap: "1rem" }}>
                    <TextField
                        variant="outlined"
                        label="Name"
                        value={name}
                        onChange={(event) => setName(event.target.value)}
                    />
                </Box>
            </DialogContent>
            <DialogActions>
                <Button onClick={props.onClose} color="inherit">Cancel</Button>
                <Button
                    disabled={!name || name.length === 0}
                    color="primary"
                    onClick={() => {
                        if (!!currentFloorData) {
                            currentFloorData.name = name;
                            setFloor(currentFloorData);
                        }
                        props.onClose();
                    }}
                >
                    Save
                </Button>
            </DialogActions>
        </Dialog>
    )
}