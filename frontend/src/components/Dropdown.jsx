import { useState, useEffect, Fragment } from 'react'
import axios from 'axios'

function DropdownSelector(
    {
        getURL,
        postURL,
        itemKey = "models",
        label = "Select Item",
        showBatchSize = false,
        onSelect
    }
) {
    //getting list of models
    const [items, setItems] = useState([]);
    const [selected, setSelected] = useState(null);
    const [batchSize, setBatchSize] = useState(30);

    useEffect(() => {
        axios.get(getURL)
            .then(res => setItems(res.data[itemKey] || []))
            .catch(err => {
                console.error(`Failed to fetch ${itemKey}:`, err);
                setItems([]);
            });
    }, [getURL, itemKey]);


    const handleSelect = async (item) => {
        setSelected(item);
        try {
            //with batchsize enabled
            if (showBatchSize) {
                await axios.post(postURL, { [itemKey]: item, batch_size: batchSize });
            } else {
                await axios.post(postURL, { [itemKey]: item });
            }
            if (onSelect) onSelect(item);
        } catch (err) {
            console.error(`Failed to select ${itemKey}:`, err);
        }
    }

    const handleBatchSizeChange = (e) => {
        setBatchSize(Number(e.target.value));
    }


    return (
        <Fragment>
            {showBatchSize && (
                <div className="w-full max-w-xs mt-4 flex flex-col items-center">
                    <label className="mb-2">Batch Size: {batchSize}</label>
                    <input
                        type="range"
                        min={1}
                        max={100}
                        value={batchSize}
                        className="range"
                        onChange={handleBatchSizeChange}
                    />
                </div>
            )}
            <div className="dropdown dropdown-bottom dropdown-center">
                <div tabIndex={0} role="button" className="btn m-1 w-52">{selected ? selected : label}</div>
                <ul tabIndex={0} className="dropdown-content menu bg-base-100 rounded-box z-1 w-52 p-2 shadow-sm">
                    {items.map(item => (
                        <li key={item}>
                            <button
                                onClick={() => {
                                    handleSelect(item);
                                    document.activeElement.blur();
                                }}>{item}</button>
                        </li>
                    ))
                    }
                </ul>
            </div>
        </Fragment>

    )
}

export default DropdownSelector