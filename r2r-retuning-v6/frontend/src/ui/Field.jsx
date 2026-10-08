export function Field({ label, value, onChange, type = 'number', step = '0.01' }) {
  return (
    <label className="field">
      <span>{label}</span>
      <input type={type} value={value} step={step} onChange={(event) => onChange(event.target.value)} />
    </label>
  );
}

export function SelectField({ label, value, onChange, options }) {
  const normalizedOptions = options.map((option) =>
    typeof option === 'string' ? { value: option, label: option } : option,
  );
  return (
    <label className="field">
      <span>{label}</span>
      <select value={value} onChange={(event) => onChange(event.target.value)}>
        {normalizedOptions.map((option) => (
          <option value={option.value} key={option.value}>
            {option.label}
          </option>
        ))}
      </select>
    </label>
  );
}
