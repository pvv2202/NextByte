import React from 'react'
import { ClipLoader } from 'react-spinners'
import { PulseDiv } from '../bot'
import { useState } from 'react';
import { Navigate, useNavigate } from 'react-router';
import DropdownMenu from '../dropdown-menu/dropdown'
import PwdInput from './pwdInput';
import countries from '../../data/country-flag.json'
import states from '../../data/states.json'
import cities from '../../data/cities.json'




function SignupForm({handleSubmit, onSubmit, register, authenticating, userError, Controller, control, watch}) {
    const password = watch('password')
    const confirm = watch('confirm-password')
    const navigate = useNavigate('')

    
    return (
        <form className="flex flex-col gap-4 bg-white p-6 rounded-xl shadow-md shadow-green-400" onSubmit={handleSubmit(onSubmit)}>
            <h2 className='flex gap-1 items-center justify-between text-gray-800 text-xl mt-4'>
                <p>Create Your Account</p>
                <PulseDiv>
                    <img className=' max-w-24 object-contain' src={'Closing-eyes.gif'} alt="nextbyte" />
                </PulseDiv>
            </h2>
            
            {userError ? <p className='text-red-600 text-sm ml-2'>{userError}</p> : null}
            <input type='text' autoComplete='off' autoFocus='off' {...register("username", {required: true,
                maxLength: 20})} 
                placeholder='Username'
                className='p-2 rounded'
            />
            
            <Controller 
                name="password"
                control={control}
                rules={{required: true, maxLength: 20}}
                render={({field}) => (
                    <PwdInput 
                        
                        value={field.value}
                        onChange={field.onChange}
                        name='Password'
                       
                    />
                )}
            />
            {password !== confirm ? <p className='text-red-600 text-sm ml-2'>Passwords must match</p> : null}
            <Controller 
                name="confirm-password"
                control={control}
                rules={{required: true, maxLength: 20}}
                render={({field}) => (
                    <PwdInput 
                        value={field.value}
                        onChange={field.onChange}
                        name='Confirm password'
                
                    />
                )}
            />

          
            <input type='number' autoComplete='off' autoFocus='off' {...register("age", {required: true})}
                placeholder='age'
                className='p-2 rounded'
            />
           
            <input type="text" autoComplete='off' autoFocus='off' {...register('email', {required: false})} 
                placeholder='Email (optional)'
                className='p-2 rounded'
            />
            
            <Controller 
                name="country"
                control={control}
                render={({field: countryField}) => (
                    <>
                        <DropdownMenu
                            value={countryField.value}
                            onChange={countryField.onChange}
                            {...countryField}
                            options={countries}
                            imgpath={['flags', 'svg']}
                            category={'Country'}
                            valueKey={'name'}
                        />
                        {countryField.value == 'United States of America' && (
                            <>
                                <Controller 
                                    name="state"
                                    control={control}
                                    render={({field: stateField}) => (
                                    <>
                                        <DropdownMenu
                                            value={stateField.value}
                                            onChange={stateField.onChange}
                                            {...stateField}
                                            options={states}
                                            imgpath={[]}
                                            category={'State'}
                                            valueKey={'name'}
                                        />
                                        {stateField.value && (
                                        <Controller 
                                            name="city"
                                            control={control}
                                            render={({field: cityField}) => (
                                            <DropdownMenu
                                                value={cityField.value}
                                                onChange={cityField.onChange}
                                                {...cityField}
                                                options={cities.filter(city => city.state == stateField.value)}
                                                imgpath={[]}
                                                category={'City'}
                                                valueKey = {'city'}
                                            />
                                            )}
                                        />
                                        )}
                                    </>
                                    )}
                                />
                                
                            </>
                            
                        )}
                    </>
                )}
            />

            
            <div className='flex justify-end'>
                <button type='submit' disabled={(password || '').length < 1 || (password !== confirm) ? true : false} className='w-20 rounded curser:hand hover:scale-105 text-white  ml-3 p-2 bg-gray-800'>
                    {authenticating ? <ClipLoader color={'white'} speedMultiplier={0.5} size={20} /> : <p>Sign up</p>}
                </button>
            </div>
           <button onClick={() => navigate('/login')}>Already have an account?</button>
        </form>
    )
}

export default SignupForm