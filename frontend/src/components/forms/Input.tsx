import React, { InputHTMLAttributes } from 'react';

interface InputProps extends InputHTMLAttributes<HTMLInputElement> {
  label?: string;
  helperText?: string;
  error?: string;
  fullWidth?: boolean;
}

const Input: React.FC<InputProps> = ({
  label,
  helperText,
  error,
  fullWidth = false,
  className = '',
  id,
  ...rest
}) => {
  const inputId = id || `input-${Math.random().toString(36).substring(2, 9)}`;

  return (
    <div className={`mb-4 ${fullWidth ? 'w-full' : ''}`}>
      {label && (
        <label
          htmlFor={inputId}
          className="block text-sm font-medium text-gray-700 mb-1"
        >
          {label}
        </label>
      )}
      <input
        id={inputId}
        className={`
          block px-3 py-2 bg-white border rounded-md shadow-sm focus:outline-none sm:text-sm
          ${error ? 'border-red-300 text-red-900 focus:ring-red-500 focus:border-red-500' : 'border-gray-300 focus:ring-blue-500 focus:border-blue-500'}
          ${fullWidth ? 'w-full' : ''}
          ${className}
        `}
        aria-invalid={!!error}
        aria-describedby={
          helperText ? `${inputId}-helper-text` : error ? `${inputId}-error` : undefined
        }
        {...rest}
      />
      {helperText && !error && (
        <p
          id={`${inputId}-helper-text`}
          className="mt-1 text-sm text-gray-500"
        >
          {helperText}
        </p>
      )}
      {error && (
        <p
          id={`${inputId}-error`}
          className="mt-1 text-sm text-red-600"
          role="alert"
        >
          {error}
        </p>
      )}
    </div>
  );
};

export default Input;