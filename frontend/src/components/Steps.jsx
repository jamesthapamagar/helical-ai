import React from 'react'

function Steps({ currentStep }) {
    const steps = ["Dataset", "Model", "Application", "Review"]
    return (
        <ul className="steps">
            {steps.map((label, index) => (
                <li
                    key={label}
                    className={
                        "step" +
                        (index < currentStep ? " step-primary" : "") +
                        (index === currentStep ? " step-primary" : "")
                    }
                >
                    {label}
                </li>
            ))}
        </ul>
    )
}

export default Steps